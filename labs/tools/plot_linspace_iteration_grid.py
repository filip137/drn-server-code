#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_iteration_grid(npz_path: Path, key: str) -> np.ndarray:
    data = np.load(npz_path)
    if key in data:
        grid = np.asarray(data[key])
    elif key == "iteration_grid" and "iteration_counts" in data:
        counts = np.asarray(data["iteration_counts"])
        n = int(round(float(np.sqrt(counts.size))))
        if n * n != counts.size:
            raise ValueError(
                f"Cannot infer square grid from iteration_counts with size {counts.size}."
            )
        grid = counts.reshape(n, n)
    else:
        raise KeyError(f"Key '{key}' not found in {npz_path}.")

    if grid.ndim == 1:
        n = int(round(float(np.sqrt(grid.size))))
        if n * n != grid.size:
            raise ValueError(f"Expected square-sized 1D array, got length {grid.size}.")
        grid = grid.reshape(n, n)
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid for key '{key}', got shape {grid.shape}.")
    return grid


def _infer_extent(run_dir: Path, expected_size: int):
    inputs_path = run_dir / "linspace_inputs.npz"
    if not inputs_path.exists():
        return None
    try:
        inputs = np.load(inputs_path)["inputs"]
    except Exception:
        return None
    inputs = np.asarray(inputs)
    if inputs.ndim != 2 or inputs.shape[1] < 2 or inputs.shape[0] != expected_size:
        return None
    x = inputs[:, 0]
    y = inputs[:, 1]
    return [float(x.min()), float(x.max()), float(y.min()), float(y.max())]


def plot_iteration_grid(npz_path: Path, key: str, output_path: Path, title: str | None) -> Path:
    grid = _load_iteration_grid(npz_path, key)
    extent = _infer_extent(npz_path.parent, grid.size)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(
        grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    if extent is None:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_title(title or f"{npz_path.parent.name} iteration counts")
    fig.colorbar(im, ax=ax, label="equilibrium iterations", pad=0.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot linspace equilibrium iteration counts as a 2D grid.")
    parser.add_argument(
        "--iteration-npz",
        required=True,
        help="Path to linspace_iteration_counts.npz.",
    )
    parser.add_argument(
        "--key",
        default="iteration_grid",
        help="NPZ key to plot (default: iteration_grid).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <run_dir>/<key>.png).",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    args = parser.parse_args(argv)

    npz_path = Path(args.iteration_npz).expanduser().resolve()
    if not npz_path.exists():
        raise SystemExit(f"File not found: {npz_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else npz_path.parent / f"{args.key}.png"
    )

    saved = plot_iteration_grid(npz_path, args.key, output_path, args.title)
    print(f"Saved iteration grid plot to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
