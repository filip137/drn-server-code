import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _layer_sort_key(name):
    try:
        return int("".join(ch for ch in name if ch.isdigit()))
    except ValueError:
        return name


def load_npz_params(npz_path):
    npz_path = Path(npz_path).expanduser()
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compute_relative_L1_error(cd_voltages, spice_voltages, axis=1, eps=1e-12):
    """
    Compute per-sample relative L1 error between CD and SPICE voltages.

    Parameters
    ----------
    cd_voltages : np.ndarray
        Array of voltages from the coordinate-descent solver.
        Shape typically [N_samples, N_nodes] (or [N_samples, N_out]).
    spice_voltages : np.ndarray
        Array of voltages from SPICE, same shape as cd_voltages.
    axis : int
        Axis over which to average (usually 1 = nodes).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    rel_err : np.ndarray
        Relative L1 error per sample, shape [N_samples].
        rel_err[i] = (mean_j |cd - spice|) / (mean_j |spice| + eps)
    mae : np.ndarray
        Mean absolute error per sample, shape [N_samples].
    ref_mean_abs : np.ndarray
        Mean absolute SPICE voltage per sample, shape [N_samples].
    """
    cd_voltages = np.asarray(cd_voltages)
    spice_voltages = np.asarray(spice_voltages)

    if cd_voltages.shape != spice_voltages.shape:
        raise ValueError(
            f"Shape mismatch: cd_voltages {cd_voltages.shape} vs "
            f"spice_voltages {spice_voltages.shape}"
        )

    diff = cd_voltages - spice_voltages
    mae = np.mean(np.abs(diff), axis=axis)              # mean |cd - spice|
    ref_mean_abs = np.mean(np.abs(spice_voltages), axis=axis)  # mean |spice|
    rel_err = mae / (ref_mean_abs + eps)

    return rel_err, mae, ref_mean_abs


def _prepare_plot_values(values, log_scale, label):
    if log_scale:
        return np.log10(np.clip(values, 1e-16, None)), f"log10 {label}"
    return values, label


def _validate_grid_values(X_grid, values):
    X_grid = np.asarray(X_grid)
    values = np.asarray(values)

    if X_grid.ndim != 2 or X_grid.shape[1] < 2:
        raise ValueError(
            f"X_grid should have shape [N_samples, 2], got {X_grid.shape}"
        )
    if X_grid.shape[0] != values.shape[0]:
        raise ValueError(
            f"Mismatch: X_grid has {X_grid.shape[0]} samples, "
            f"values has {values.shape[0]}."
        )
    return X_grid, values


def _scatter_grid(
    ax,
    X_grid,
    values,
    log_scale=True,
    s=10,
    label="value",
    vmin=None,
    vmax=None,
    x_label="Input x",
    y_label="Input y",
):
    X_grid, values = _validate_grid_values(X_grid, values)
    plot_values, cb_label = _prepare_plot_values(values, log_scale, label)
    sc = ax.scatter(X_grid[:, 0], X_grid[:, 1], c=plot_values, s=s, vmin=vmin, vmax=vmax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return sc, cb_label


def plot_input_error(
    X_grid,
    error,
    ax=None,
    title=None,
    log_scale=True,
    s=10,
    label="relative L1 error",
    vmin=None,
    vmax=None,
    add_colorbar=True,
    x_label="Input x",
    y_label="Input y",
):
    """
    Plot input grid colored by an error metric (e.g. relative L1 error).

    Parameters
    ----------
    X_grid : np.ndarray
        Input coordinates, shape [N_samples, 2] (columns = x, y).
    error : np.ndarray
        Error per sample, shape [N_samples].
    ax : matplotlib.axes.Axes or None
        Existing axis to plot on. If None, a new figure and axis are created.
    title : str or None
        Optional title for the plot.
    log_scale : bool
        If True, plot log10(error) instead of error.
    s : float
        Marker size for the scatter plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots()

    sc, cb_label = _scatter_grid(
        ax,
        X_grid,
        error,
        log_scale=log_scale,
        s=s,
        label=label,
        vmin=vmin,
        vmax=vmax,
        x_label=x_label,
        y_label=y_label,
    )
    if add_colorbar:
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(cb_label)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def save_grid_plot(
    grid,
    values,
    output_dir,
    filename,
    title,
    label,
    log_scale,
    s=12,
    x_label="Input x",
    y_label="Input y",
):
    ax = plot_input_error(
        grid,
        values,
        title=title,
        log_scale=log_scale,
        s=s,
        label=label,
        x_label=x_label,
        y_label=y_label,
    )
    fig_path = output_dir / filename
    ax.figure.savefig(fig_path, dpi=200)
    plt.close(ax.figure)
    print(f"Saved scatter plot to {fig_path}")
    return fig_path


def _log_suffix(log_scale):
    return "_log" if log_scale else ""


def _safe_label(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _parse_layer_labels(items):
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Layer label must be in the form Layer_1=Label, got: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def resolve_paths(cd_path, spice_path, base_dir):
    """
    Resolve CD/SPICE voltage file paths. If either path is None, fall back to the
    first matching file in base_dir using simple glob patterns.
    """
    if cd_path is not None and spice_path is not None:
        return Path(cd_path).expanduser(), Path(spice_path).expanduser()

    base_dir = Path(base_dir).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    cd_candidates = sorted(base_dir.glob("*cd*.npz"))
    spice_candidates = sorted(base_dir.glob("*spice*.npz"))
    if not cd_candidates:
        raise FileNotFoundError(f"No *cd*.npz files found in {base_dir}")
    if not spice_candidates:
        raise FileNotFoundError(f"No *spice*.npz files found in {base_dir}")
    return cd_candidates[0], spice_candidates[0]


def extract_input_grid(cd_voltages):
    """Recover the original 2D input grid from the recorded Layer_0 states if present."""
    layer0 = cd_voltages.get("Layer_0")
    if layer0 is None or layer0.ndim != 2 or layer0.shape[1] != 2:
        return None
    return layer0


def build_index_grid(n_samples):
    """Fallback grid when Layer_0 is missing: plot against a synthetic sigma grid."""
    side = int(np.sqrt(n_samples))
    if side * side == n_samples:
        rows, cols = side, side
    else:
        rows = max(1, side)
        cols = int(np.ceil(n_samples / rows))
        if rows * cols < n_samples:
            rows += 1

    x = np.linspace(-3.0, 3.0, cols)
    y = np.linspace(-3.0, 3.0, rows)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel()[:n_samples], yy.ravel()[:n_samples]))
    return grid


def _reduce_to_samples(values):
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError("Expected array-like values with at least 1 dimension.")
    if values.ndim > 1:
        reduce_axes = tuple(range(1, values.ndim))
        values = np.mean(values, axis=reduce_axes)
    return values


def _flatten_samples(values):
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError("Expected array-like values with at least 1 dimension.")
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def _extract_iter_value(path_str):
    match = re.search(r"iter(\d+)", path_str)
    if not match:
        return None
    return int(match.group(1))


def plot_npz_grid(
    npz_path,
    output_dir=None,
    log_scale=False,
    keys=None,
    label="value",
    x_label="Input x",
    y_label="Input y",
):
    """
    Plot per-sample values stored in a .npz file on a 2D grid.
    """
    data = load_npz_params(npz_path)
    output_dir = prepare_output_dir(output_dir, npz_path)
    plot_keys = list(data.keys()) if keys is None else list(keys)
    safe_label = label.replace(" ", "_").replace("/", "_")
    suffix = _log_suffix(log_scale)

    for key in plot_keys:
        if key not in data:
            raise KeyError(f"Key {key} not found in {npz_path}")
        values = _reduce_to_samples(data[key])
        grid = build_index_grid(values.shape[0])
        save_grid_plot(
            grid,
            values,
            output_dir,
            f"{key}_{safe_label}{suffix}.png",
            f"{key} {label}",
            label,
            log_scale,
            s=12,
            x_label=x_label,
            y_label=y_label,
        )


def plot_mean_curve(
    npz_paths,
    key,
    output_dir=None,
    iter_values=None,
    label="mean residual",
    log_scale=False,
    filename_suffix="",
):
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    if not npz_paths:
        raise ValueError("npz_paths must contain at least one path.")

    if iter_values is None:
        iter_values = []
        for path in npz_paths:
            guess = _extract_iter_value(str(path))
            if guess is None:
                raise ValueError(
                    "Could not infer iteration values from paths; pass --mean-curve-iters."
                )
            iter_values.append(guess)
    if len(iter_values) != len(npz_paths):
        raise ValueError("iter_values length must match npz_paths.")

    means = []
    for path in npz_paths:
        data = load_npz_params(path)
        if key not in data:
            raise KeyError(f"Key {key} not found in {path}")
        values = _reduce_to_samples(data[key])
        means.append(float(np.mean(values)))

    output_dir = prepare_output_dir(output_dir, npz_paths[0])
    fig, ax = plt.subplots()
    if log_scale:
        means_plot = np.log10(np.clip(means, 1e-16, None))
        y_label = f"log10 {label}"
    else:
        means_plot = means
        y_label = label
    ax.plot(iter_values, means_plot, marker="o")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.set_title(f"{key} mean vs iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_label = label.replace(" ", "_").replace("/", "_")
    suffix = _log_suffix(log_scale)
    extra = f"_{filename_suffix}" if filename_suffix else ""
    fig_path = output_dir / f"{key}_{safe_label}_mean_vs_iters{suffix}{extra}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved mean curve to {fig_path}")
    return fig_path


def plot_npz_grid_comparison(
    npz_paths,
    key,
    output_dir=None,
    log_scale=False,
    label="value",
    titles=None,
    cols=2,
    x_label="Input x",
    y_label="Input y",
):
    """
    Plot the same key from multiple .npz files in a single multi-panel figure.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    if not npz_paths:
        raise ValueError("npz_paths must contain at least one path.")

    output_dir = prepare_output_dir(output_dir, npz_paths[0])
    values_list = []
    grids = []
    keys = list(key) if isinstance(key, (list, tuple)) else [key] * len(npz_paths)
    if len(keys) != len(npz_paths):
        raise ValueError("grid-compare keys must match the number of npz paths.")

    for path, key_name in zip(npz_paths, keys):
        data = load_npz_params(path)
        if key_name not in data:
            raise KeyError(f"Key {key_name} not found in {path}")
        values = _reduce_to_samples(data[key_name])
        values_list.append(values)
        grids.append(build_index_grid(values.shape[0]))

    plot_values = [_prepare_plot_values(values, log_scale, label)[0] for values in values_list]
    vmin = min(np.min(vals) for vals in plot_values)
    vmax = max(np.max(vals) for vals in plot_values)

    n_plots = len(npz_paths)
    cols = max(1, min(cols, n_plots))
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2), squeeze=False)

    cb_label = None
    last_sc = None
    for idx, (path, key_name, values, grid) in enumerate(zip(npz_paths, keys, values_list, grids)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        last_sc, cb_label = _scatter_grid(
            ax,
            grid,
            values,
            log_scale=log_scale,
            s=12,
            label=label,
            vmin=vmin,
            vmax=vmax,
            x_label=x_label,
            y_label=y_label,
        )
        if titles and idx < len(titles):
            title = titles[idx]
        else:
            path_obj = Path(path)
            title = path_obj.parent.name or path_obj.stem
        ax.set_title(f"{title} {key_name}")

    for idx in range(n_plots, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    if last_sc is not None and cb_label:
        fig.colorbar(last_sc, ax=axes.ravel().tolist(), label=cb_label, shrink=0.9)

    fig.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "_")
    suffix = _log_suffix(log_scale)
    key_label = keys[0] if len(set(keys)) == 1 else "mixed_keys"
    fig_path = output_dir / f"{key_label}_{safe_label}_comparison{suffix}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot to {fig_path}")
    return fig_path


def plot_npz_error_comparison(
    npz_paths,
    keys=None,
    output_dir=None,
    log_scale=False,
    label="relative L1 error",
    x_label="Input x",
    y_label="Input y",
):
    """Compute per-sample error between two npz files and plot error on a grid."""
    if len(npz_paths) != 2:
        raise ValueError("error-compare requires exactly two npz paths.")

    data_a = load_npz_params(npz_paths[0])
    data_b = load_npz_params(npz_paths[1])

    if keys is None:
        common = sorted(set(data_a) & set(data_b), key=_layer_sort_key)
        if common:
            key_a = key_b = common[-1]
        else:
            key_a = sorted(data_a.keys(), key=_layer_sort_key)[-1]
            key_b = sorted(data_b.keys(), key=_layer_sort_key)[-1]
            print(f"No common keys; using {key_a} vs {key_b}")
    else:
        keys = list(keys) if isinstance(keys, (list, tuple)) else [keys, keys]
        if len(keys) == 1:
            key_a = key_b = keys[0]
        elif len(keys) == 2:
            key_a, key_b = keys
        else:
            raise ValueError("error-compare keys must contain one or two entries.")

    if key_a not in data_a:
        raise KeyError(f"Key {key_a} not found in {npz_paths[0]}")
    if key_b not in data_b:
        raise KeyError(f"Key {key_b} not found in {npz_paths[1]}")

    values_a = _flatten_samples(data_a[key_a])
    values_b = _flatten_samples(data_b[key_b])
    rel_err, mae, ref_mean_abs = compute_relative_L1_error(values_a, values_b, axis=1)

    if output_dir is None:
        parents = [Path(p).expanduser().resolve().parent for p in npz_paths]
        common_dir = Path(os.path.commonpath(parents))
        output_dir = prepare_output_dir(common_dir, npz_paths[0])
    else:
        output_dir = prepare_output_dir(output_dir, npz_paths[0])
    grid = extract_input_grid(data_a)
    if grid is None:
        grid = extract_input_grid(data_b)
    if grid is None:
        grid = build_index_grid(rel_err.shape[0])
    safe_label = label.replace(" ", "_").replace("/", "_")
    suffix = _log_suffix(log_scale)

    rel_path = save_grid_plot(
        grid,
        rel_err,
        output_dir,
        f"{key_a}_vs_{key_b}_relative_L1_error{suffix}.png",
        f"{key_a} vs {key_b} relative L1 error",
        label,
        log_scale,
        x_label=x_label,
        y_label=y_label,
    )
    mae_path = save_grid_plot(
        grid,
        mae,
        output_dir,
        f"{key_a}_vs_{key_b}_absolute_error{suffix}.png",
        f"{key_a} vs {key_b} mean absolute error",
        "mean absolute error",
        log_scale,
        x_label=x_label,
        y_label=y_label,
    )
    summary_path = output_dir / f"{key_a}_vs_{key_b}_error_summary{suffix}.json"
    summary_path.write_text(json.dumps(summarize_errors(rel_err, mae, ref_mean_abs), indent=2))
    print(f"Saved error summary to {summary_path}")
    print(f"Saved error plots to {rel_path} and {mae_path}")
    return summary_path


def write_npz_error_summary(
    npz_paths,
    keys=None,
    output_dir=None,
    log_scale=False,
):
    """Compute per-sample error between two npz files and write summary JSON (no plots)."""
    if len(npz_paths) != 2:
        raise ValueError("error-compare requires exactly two npz paths.")

    data_a = load_npz_params(npz_paths[0])
    data_b = load_npz_params(npz_paths[1])

    if keys is None:
        common = sorted(set(data_a) & set(data_b), key=_layer_sort_key)
        if common:
            key_a = key_b = common[-1]
        else:
            key_a = sorted(data_a.keys(), key=_layer_sort_key)[-1]
            key_b = sorted(data_b.keys(), key=_layer_sort_key)[-1]
            print(f"No common keys; using {key_a} vs {key_b}")
    else:
        keys = list(keys) if isinstance(keys, (list, tuple)) else [keys, keys]
        if len(keys) == 1:
            key_a = key_b = keys[0]
        elif len(keys) == 2:
            key_a, key_b = keys
        else:
            raise ValueError("error-compare keys must contain one or two entries.")

    if key_a not in data_a:
        raise KeyError(f"Key {key_a} not found in {npz_paths[0]}")
    if key_b not in data_b:
        raise KeyError(f"Key {key_b} not found in {npz_paths[1]}")

    values_a = _flatten_samples(data_a[key_a])
    values_b = _flatten_samples(data_b[key_b])
    rel_err, mae, ref_mean_abs = compute_relative_L1_error(values_a, values_b, axis=1)

    if output_dir is None:
        parents = [Path(p).expanduser().resolve().parent for p in npz_paths]
        common_dir = Path(os.path.commonpath(parents))
        output_dir = prepare_output_dir(common_dir, npz_paths[0])
    else:
        output_dir = prepare_output_dir(output_dir, npz_paths[0])

    suffix = _log_suffix(log_scale)
    summary_path = output_dir / f"{key_a}_vs_{key_b}_error_summary{suffix}.json"
    summary_path.write_text(json.dumps(summarize_errors(rel_err, mae, ref_mean_abs), indent=2))
    print(f"Saved error summary to {summary_path}")
    return summary_path


def summarize_errors(rel_err, mae, ref_mean_abs):
    return {
        "mean_rel_l1": float(np.mean(rel_err)),
        "median_rel_l1": float(np.median(rel_err)),
        "max_rel_l1": float(np.max(rel_err)),
        "mean_mae": float(np.mean(mae)),
        "max_mae": float(np.max(mae)),
        "mean_abs_spice": float(np.mean(ref_mean_abs)),
    }


def write_percentile_error_summaries(
    cd_npz,
    spice_npz,
    output_dir=None,
    percentiles=(50, 90, 99),
    keys=None,
):
    """
    Write percentile summaries (median/P90/P99 by default) for relative L1 error and MAE.

    Creates one folder per percentile (e.g., median, p90, p99) with an error_summary.json.
    """
    cd_data = load_npz_params(cd_npz)
    spice_data = load_npz_params(spice_npz)

    if keys is None:
        layers = sorted(set(cd_data) & set(spice_data), key=_layer_sort_key)
        if not layers:
            raise ValueError("No common layers found between the two files.")
    else:
        layers = list(keys) if isinstance(keys, (list, tuple)) else [keys]
        missing = [layer for layer in layers if layer not in cd_data or layer not in spice_data]
        if missing:
            raise KeyError(f"Missing layers in inputs: {', '.join(missing)}")

    if output_dir is None:
        parents = [Path(p).expanduser().resolve().parent for p in (cd_npz, spice_npz)]
        common_dir = Path(os.path.commonpath(parents))
        output_dir = prepare_output_dir(common_dir, cd_npz)
    else:
        output_dir = prepare_output_dir(output_dir, cd_npz)

    for pct in percentiles:
        label = "median" if int(pct) == 50 else f"p{int(pct)}"
        out_dir = output_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "percentile": float(pct),
            "cd_file": str(Path(cd_npz).expanduser()),
            "spice_file": str(Path(spice_npz).expanduser()),
            "layers": {},
        }
        for layer in layers:
            rel_err, mae, _ = compute_relative_L1_error(cd_data[layer], spice_data[layer])
            summary["layers"][layer] = {
                "rel_l1": float(np.percentile(rel_err, pct)),
                "mae": float(np.percentile(mae, pct)),
            }
        (out_dir / "error_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return output_dir


def prepare_output_dir(output_dir, cd_voltages_path):
    """
    Create the output directory, falling back to a local folder if the preferred
    location is not writable (e.g., outside the workspace sandbox).
    """
    preferred = Path(output_dir) if output_dir else Path(cd_voltages_path).expanduser().parent
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test_path = preferred / ".write_test"
        test_path.touch()
        test_path.unlink()
        return preferred
    except PermissionError:
        fallback = Path.cwd() / "paper_post_processing_outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"Permission denied writing to {preferred}; using {fallback} instead.")
        return fallback


def main(
    cd_voltages_path,
    spice_voltages_path,
    output_dir=None,
    log_scale=False,
    plot_layer=None,
    neuron_source="cd",
    neuron_index=None,
    random_seed=None,
    layer_label_map=None,
    x_label="Input x",
    y_label="Input y",
    save_error_data=False,
):
    cd_voltages = load_npz_params(cd_voltages_path)
    spice_voltages = load_npz_params(spice_voltages_path)

    common_layers = sorted(set(cd_voltages) & set(spice_voltages), key=_layer_sort_key)
    if not common_layers:
        raise ValueError(
            f"No overlapping layers between CD ({list(cd_voltages)}) "
            f"and SPICE ({list(spice_voltages)})."
        )

    output_dir = prepare_output_dir(output_dir, cd_voltages_path)

    # Plot all layers by default; if plot_layer is provided, restrict to that.
    plot_targets = [plot_layer] if plot_layer else list(common_layers)
    input_grid = extract_input_grid(cd_voltages)
    summary = {
        "cd_file": str(Path(cd_voltages_path).expanduser()),
        "spice_file": str(Path(spice_voltages_path).expanduser()),
        "layers": {},
    }

    index_grid_warned = False
    rng = np.random.default_rng(random_seed)
    layer_label_map = layer_label_map or {}
    for layer in common_layers:
        display_layer = layer_label_map.get(layer, layer)
        safe_layer = _safe_label(display_layer)
        rel_err, mae, ref_mean_abs = compute_relative_L1_error(
            cd_voltages[layer], spice_voltages[layer]
        )
        stats = summarize_errors(rel_err, mae, ref_mean_abs)
        summary["layers"][display_layer] = stats

        print(
            f"{display_layer}: relL1 mean={stats['mean_rel_l1']:.4g}, "
            f"median={stats['median_rel_l1']:.4g}, max={stats['max_rel_l1']:.4g}; "
            f"MAE mean={stats['mean_mae']:.4g}, max={stats['max_mae']:.4g}"
        )

        if save_error_data:
            grid_data = input_grid if input_grid is not None else build_index_grid(rel_err.shape[0])
            npz_path = output_dir / f"{safe_layer}_error_data.npz"
            np.savez(
                npz_path,
                input_grid=grid_data,
                rel_err=rel_err,
                mae=mae,
                mean_abs_spice=ref_mean_abs,
            )
            print(f"Saved error data to {npz_path}")

        if layer in plot_targets:
            if plt is None:
                print("matplotlib is not installed; skipping scatter plot.")
                continue
            plot_grid = input_grid
            if plot_grid is None:
                plot_grid = build_index_grid(rel_err.shape[0])
                if not index_grid_warned:
                    rows = int(plot_grid[:, 1].max())
                    cols = int(plot_grid[:, 0].max())
                    print(
                        f"Layer_0 grid missing; plotting on synthetic {rows}x{cols} grid "
                        "over [-3, 3] in sigma units."
                    )
                    index_grid_warned = True
            suffix = _log_suffix(log_scale)
            save_grid_plot(
                plot_grid,
                rel_err,
                output_dir,
                f"{safe_layer}_relative_L1_error{suffix}.png",
                f"{display_layer} relative L1 error",
                "relative L1 error",
                log_scale,
                x_label=x_label,
                y_label=y_label,
            )
            save_grid_plot(
                plot_grid,
                mae,
                output_dir,
                f"{safe_layer}_absolute_error{suffix}.png",
                f"{display_layer} mean absolute error",
                "mean absolute error",
                log_scale,
                x_label=x_label,
                y_label=y_label,
            )

            source_map = {"cd": cd_voltages, "spice": spice_voltages}
            source_key = neuron_source.lower()
            if source_key not in source_map:
                raise ValueError(f"Unknown neuron source: {neuron_source}")
            source_voltages = source_map[source_key][layer]
            n_nodes = source_voltages.shape[1]
            if neuron_index is None:
                rand_idx = int(rng.integers(0, n_nodes))
            else:
                rand_idx = int(neuron_index)
                if not 0 <= rand_idx < n_nodes:
                    raise ValueError(
                        f"neuron_index {rand_idx} is out of range for {layer} "
                        f"(0..{n_nodes - 1})"
                    )
            neuron_values = source_voltages[:, rand_idx]
            summary["layers"][display_layer]["random_neuron_index"] = rand_idx
            summary["layers"][display_layer]["random_neuron_source"] = source_key

            save_grid_plot(
                plot_grid,
                neuron_values,
                output_dir,
                f"{safe_layer}_{source_key}_neuron_{rand_idx}.png",
                f"{display_layer} {source_key} neuron {rand_idx} values",
                f"{source_key} neuron value",
                log_scale=False,
                x_label=x_label,
                y_label=y_label,
            )

    summary_path = output_dir / "layer_error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare CD vs SPICE voltages and compute per-layer relative errors."
    )
    parser.add_argument("--cd", type=str, help="Path to *_cd*.npz file.")
    parser.add_argument("--spice", type=str, help="Path to *_spice*.npz file.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/filip/paper_maxicao_simulations/full_folder",
        help="Folder containing CD/SPICE .npz files (used if --cd/--spice are omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/filip/paper_simulation_results_",
        help="Where to write plots/summary.",
    )
    parser.add_argument(
        "--plot-layer",
        type=str,
        help="Layer name to use for the scatter plot (defaults to last common layer).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=False,
        help="Use log10 scale for the scatter plot colorbar.",
    )
    parser.add_argument("--grid-npz", type=str, help="Plot values from a .npz file on a 2D grid.")
    parser.add_argument(
        "--grid-compare-npz",
        nargs="+",
        help="Plot the same key from multiple .npz files in one figure.",
    )
    parser.add_argument(
        "--grid-compare-key",
        type=str,
        help="Key to plot for --grid-compare-npz.",
    )
    parser.add_argument(
        "--grid-compare-keys",
        nargs="+",
        help="Per-file keys to plot for --grid-compare-npz (one key per npz).",
    )
    parser.add_argument(
        "--grid-compare-cols",
        type=int,
        default=2,
        help="Number of columns for --grid-compare-npz layout.",
    )
    parser.add_argument(
        "--error-compare-npz",
        nargs=2,
        help="Compute error between two npz files and plot error grid (cd then spice).",
    )
    parser.add_argument(
        "--error-compare-keys",
        nargs="+",
        help="Optional key (or two keys) to compare for --error-compare-npz.",
    )
    parser.add_argument(
        "--layer-label",
        action="append",
        help="Rename a layer for display, e.g. --layer-label Layer_1=Hidden Layer",
    )
    parser.add_argument(
        "--x-label",
        default="Input x",
        help="X-axis label for grid plots.",
    )
    parser.add_argument(
        "--y-label",
        default="Input y",
        help="Y-axis label for grid plots.",
    )
    parser.add_argument(
        "--error-output-dir",
        type=str,
        default=None,
        help="Where to write error plots/summary (default: common parent of inputs).",
    )
    parser.add_argument(
        "--mean-curve-npz",
        nargs="+",
        help="Plot mean values from multiple .npz files vs iteration count.",
    )
    parser.add_argument(
        "--mean-curve-key",
        type=str,
        help="Key to use for --mean-curve-npz.",
    )
    parser.add_argument(
        "--mean-curve-iters",
        nargs="+",
        type=int,
        help="Explicit iteration values for --mean-curve-npz.",
    )
    parser.add_argument(
        "--mean-curve-suffix",
        type=str,
        default="",
        help="Optional suffix for the mean-curve output filename.",
    )
    parser.add_argument(
        "--grid-keys",
        nargs="*",
        default=None,
        help="Optional keys to plot from --grid-npz (default: all).",
    )
    parser.add_argument(
        "--grid-label",
        type=str,
        default="value",
        help="Colorbar label for --grid-npz plots.",
    )
    parser.add_argument(
        "--neuron-source",
        choices=["cd", "spice"],
        default="cd",
        help="Which voltages to sample for the random neuron plot.",
    )
    parser.add_argument(
        "--neuron-index",
        type=int,
        help="Optional neuron index to plot (defaults to random).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for selecting the neuron index.",
    )
    parser.add_argument(
        "--save-error-data",
        action="store_true",
        help="Save per-layer error arrays as .npz for replotting.",
    )
    args = parser.parse_args()

    if args.error_compare_npz:
        plot_npz_error_comparison(
            args.error_compare_npz,
            keys=args.error_compare_keys,
            output_dir=args.error_output_dir,
            log_scale=args.log_scale,
            label=args.grid_label,
            x_label=args.x_label,
            y_label=args.y_label,
        )
        raise SystemExit(0)

    if args.grid_compare_npz:
        if args.grid_compare_key and args.grid_compare_keys:
            raise ValueError("Use only one of --grid-compare-key or --grid-compare-keys")
        if not args.grid_compare_key and not args.grid_compare_keys:
            raise ValueError("--grid-compare-key or --grid-compare-keys is required with --grid-compare-npz")
        compare_key = args.grid_compare_keys if args.grid_compare_keys else args.grid_compare_key
        plot_npz_grid_comparison(
            args.grid_compare_npz,
            compare_key,
            output_dir=args.output_dir,
            log_scale=args.log_scale,
            label=args.grid_label,
            cols=args.grid_compare_cols,
            x_label=args.x_label,
            y_label=args.y_label,
        )
        raise SystemExit(0)

    if args.mean_curve_npz:
        if not args.mean_curve_key:
            raise ValueError("--mean-curve-key is required with --mean-curve-npz")
        plot_mean_curve(
            args.mean_curve_npz,
            args.mean_curve_key,
            output_dir=args.output_dir,
            iter_values=args.mean_curve_iters,
            label=args.grid_label,
            log_scale=args.log_scale,
            filename_suffix=args.mean_curve_suffix,
        )
        raise SystemExit(0)

    if args.grid_npz:
        plot_npz_grid(
            args.grid_npz,
            output_dir=args.output_dir,
            log_scale=args.log_scale,
            keys=args.grid_keys,
            label=args.grid_label,
            x_label=args.x_label,
            y_label=args.y_label,
        )
        raise SystemExit(0)

    cd_path, spice_path = resolve_paths(args.cd, args.spice, args.base_dir)
    main(
        cd_path,
        spice_path,
        output_dir=args.output_dir,
        log_scale=args.log_scale,
        plot_layer=args.plot_layer,
        neuron_source=args.neuron_source,
        neuron_index=args.neuron_index,
        random_seed=args.random_seed,
        layer_label_map=_parse_layer_labels(args.layer_label),
        x_label=args.x_label,
        y_label=args.y_label,
        save_error_data=args.save_error_data,
    )
