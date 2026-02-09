#!/usr/bin/env python3
import argparse
from pathlib import Path
import textwrap

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _layer_sort_key(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _iter_dirs(root: Path, iters: list[str] | None) -> list[Path]:
    if iters:
        dirs = []
        for it in iters:
            name = f"iter{it}" if it.isdigit() else it
            candidate = root / name
            if candidate.exists():
                dirs.append(candidate)
        return sorted(dirs)
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("iter")])


def _build_grid(inputs: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = inputs[:, 0]
    y = inputs[:, 1]
    xs = np.unique(x)
    ys = np.unique(y)
    xi = np.searchsorted(xs, x)
    yi = np.searchsorted(ys, y)
    grid = np.full((len(xs), len(ys)), np.nan, dtype=float)
    grid[xi, yi] = values
    return xs, ys, grid


def _plot_grid(xs, ys, grid, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(
        grid.T,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    wrapped_title = "\n".join(textwrap.wrap(title, width=45)) if len(title) > 45 else title
    ax.set_title(wrapped_title, pad=8)
    fig.colorbar(im, ax=ax, label="residual current (batch-avg)", pad=0.02)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_run(
    run_dir: Path,
    *,
    layer: str | None,
    all_layers: bool,
    output_root: Path,
    iter_name: str,
    iter_dir: Path,
) -> None:
    inputs_path = run_dir / "linspace_inputs.npz"
    residuals_path = run_dir / "linspace_residual_currents.npz"
    if not inputs_path.exists() or not residuals_path.exists():
        return

    inputs = np.load(inputs_path)["inputs"]
    residuals = np.load(residuals_path)
    keys = sorted(residuals.files, key=_layer_sort_key)
    if not keys:
        return

    if all_layers:
        layers = keys
    elif layer:
        if layer not in keys:
            raise SystemExit(f"Layer '{layer}' not found in {residuals_path}")
        layers = [layer]
    else:
        layers = [keys[-1]]

    n_inputs = inputs.shape[0]
    n_res = residuals[keys[0]].shape[0]
    if n_inputs % n_res != 0:
        raise SystemExit(f"Cannot map residuals to inputs: {n_inputs} inputs vs {n_res} residuals")
    batch_size = n_inputs // n_res

    for lyr in layers:
        res = residuals[lyr]
        values = np.repeat(res, batch_size)
        xs, ys, grid = _build_grid(inputs, values)
        try:
            rel_dir = run_dir.relative_to(iter_dir)
        except ValueError:
            rel_dir = Path(run_dir.name)
        out_dir = output_root / iter_name / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"residual_{lyr}.png"
        title = f"{iter_name} {run_dir.name} {lyr}"
        _plot_grid(xs, ys, grid, title, out_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Plot linspace residual currents on the 2D input grid.",
    )
    p.add_argument(
        "--root",
        required=False,
        help="Path containing iter* folders (e.g. .../hidden_3/non_amplified).",
    )
    p.add_argument("--run-dir", default=None, help="Path to a single run directory to plot.")
    p.add_argument(
        "--residuals-npz",
        default=None,
        help="Path to linspace_residual_currents.npz (inputs inferred from same folder).",
    )
    p.add_argument("--layer", default=None, help="Layer key to plot (e.g. Layer_4).")
    p.add_argument("--all-layers", action="store_true", help="Plot all layers in each run.")
    p.add_argument(
        "--iters",
        nargs="+",
        default=None,
        help="Optional iter list (e.g. 4 8 16 or iter32 iter64).",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Root folder for plots (default: <root>).",
    )
    args = p.parse_args(argv)

    if args.residuals_npz or args.run_dir:
        if args.root:
            raise SystemExit("Use --run-dir/--residuals-npz instead of --root.")
        run_dir = (
            Path(args.run_dir).expanduser().resolve()
            if args.run_dir
            else Path(args.residuals_npz).expanduser().resolve().parent
        )
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else run_dir
        )
        output_root.mkdir(parents=True, exist_ok=True)
        plot_run(
            run_dir,
            layer=args.layer,
            all_layers=args.all_layers,
            output_root=output_root,
            iter_name="run",
            iter_dir=run_dir.parent,
        )
        return 0

    if not args.root:
        raise SystemExit("--root is required unless --run-dir or --residuals-npz is provided.")

    root = Path(args.root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    iter_dirs = _iter_dirs(root, args.iters)
    if not iter_dirs:
        raise SystemExit(f"No iter* directories found under {root}")

    for iter_dir in iter_dirs:
        runs = sorted(iter_dir.rglob("linspace_inputs.npz"))
        for inputs_path in runs:
            run_dir = inputs_path.parent
            plot_run(
                run_dir,
                layer=args.layer,
                all_layers=args.all_layers,
                output_root=output_root,
                iter_name=iter_dir.name,
                iter_dir=iter_dir,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
