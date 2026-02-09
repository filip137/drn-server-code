from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from labs.mnist_tests import PcaGridSpec, prepare_mnist, sweep_pca


def _infer_grid_size(loader) -> int:
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "__len__"):
        return len(loader.dataset)
    return -1


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test PCA sweep outputs.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--model-key", required=True, help="Model key under config['models'].")
    parser.add_argument("--weights", default=None, help="Optional .pt weights path.")
    parser.add_argument("--num-iterations", type=int, default=2, help="Minimizer iterations.")
    parser.add_argument("--pca-steps", type=int, default=5, help="Grid steps per axis.")
    parser.add_argument("--pca-n-sigma", type=float, default=1.0, help="Grid extent in stddevs.")
    parser.add_argument("--pca-batch-size", type=int, default=10, help="Batch size for PCA grid.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    parts = prepare_mnist(
        config_path=args.config,
        model_key=args.model_key,
        weights_path=args.weights,
        data_mode="pca_grid",
        pca=PcaGridSpec(
            steps=args.pca_steps,
            n_sigma=args.pca_n_sigma,
            batch_size=args.pca_batch_size,
            include_idx=False,
        ),
        verbose=False,
    )

    grid_size = _infer_grid_size(parts.test_loader)
    if grid_size <= 0:
        raise SystemExit("Failed to infer PCA grid size from loader.")

    stats = sweep_pca(parts, args.num_iterations, output="stats", verbose=False)
    if not stats:
        raise SystemExit("stats snapshot is empty.")

    residuals = sweep_pca(parts, args.num_iterations, output="residuals", verbose=False)
    if not residuals:
        raise SystemExit("residual currents are empty.")
    for name, values in residuals.items():
        if not values:
            raise SystemExit(f"residual currents empty for layer {name}.")

    states = sweep_pca(parts, args.num_iterations, output="states", verbose=False)
    if not states:
        raise SystemExit("layer states are empty.")
    for name, tensor in states.items():
        if tensor.shape[0] != grid_size:
            raise SystemExit(
                f"layer {name} has {tensor.shape[0]} samples, expected {grid_size}."
            )

    print("smoke_pca_sweep: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
