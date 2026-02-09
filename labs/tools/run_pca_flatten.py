#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/filip/server_code")
LABS_DIR = PROJECT_ROOT / "labs"
FUNCTIONS_DIR = Path(__file__).resolve().parent
for path in (FUNCTIONS_DIR, PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common import export_pt_to_npz  # noqa: E402
from mnist_tests import PcaGridSpec, prepare_mnist, sweep_pca  # noqa: E402
from pt_npz import _flatten_weights_npz, _flatten_npz  # noqa: E402


def _unique_dir(base_dir: Path, stem: str) -> Path:
    candidate = base_dir / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    for idx in range(1, 1000):
        candidate = base_dir / f"{stem}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Could not create unique run dir under {base_dir}")


def run_pca_flatten(
    *,
    config_path: str,
    model_key: str,
    weights_pt: str,
    num_iterations: int,
    pca_steps: int = 30,
    pca_n_sigma: float = 3.0,
    pca_batch_size: int | None = None,
    input_layer: str | None = None,
    output_root: str = "/home/filip/paper_simulation_results_/cases",
):
    output_root = Path(output_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_dir(output_root, f"{timestamp}_pca_sweep")

    config_path = Path(config_path).expanduser().resolve()
    config_copy = run_dir / config_path.name
    shutil.copy2(config_path, config_copy)

    weights_path = Path(weights_pt).expanduser().resolve()
    run_info = {
        "timestamp": timestamp,
        "config_path": str(config_path),
        "model_key": model_key,
        "weights_path": str(weights_path),
        "num_iterations": num_iterations,
        "pca_steps": pca_steps,
        "pca_n_sigma": pca_n_sigma,
        "pca_batch_size": pca_batch_size,
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    parts = prepare_mnist(
        config_path=str(config_path),
        model_key=model_key,
        weights_path=str(weights_path),
        data_mode="pca_grid",
        pca=PcaGridSpec(
            steps=pca_steps,
            n_sigma=pca_n_sigma,
            batch_size=pca_batch_size,
        ),
    )
    layer_states = sweep_pca(parts, num_iterations)
    pca_out = run_dir / f"pca_sweep_iter{num_iterations}.npz"
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in layer_states.items()}
    np.savez(pca_out, **arrays)

    weights_npz = run_dir / "model.npz"
    export_pt_to_npz(weights_path, weights_npz)
    weights_flat, shapes_path = _flatten_weights_npz(weights_npz, run_dir)

    inputs_flat, input_shapes = _flatten_npz(pca_out, run_dir, input_layer=input_layer)
    if shapes_path.exists():
        shapes = json.loads(shapes_path.read_text())
    else:
        shapes = {}
    shapes.update(input_shapes)
    shapes_path.write_text(json.dumps(shapes, indent=2))

    return {
        "run_dir": run_dir,
        "config_copy": config_copy,
        "pca_out": pca_out,
        "weights_npz": weights_npz,
        "weights_flat": weights_flat,
        "inputs_flat": inputs_flat,
        "shapes_path": shapes_path,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="run_pca_flatten.py",
        description="Run PCA sweep, then flatten weights + inputs into a cases folder.",
    )
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--model-key", required=True, help="Model key in config['models'].")
    p.add_argument("--weights-pt", required=True, help="Path to model weights (.pt).")
    p.add_argument("--num-iterations", type=int, required=True, help="Minimizer iterations for PCA sweep.")
    p.add_argument("--pca-steps", type=int, default=30, help="Grid steps per PCA axis.")
    p.add_argument("--pca-n-sigma", type=float, default=3.0, help="Extent of PCA sweep in stddevs.")
    p.add_argument("--pca-batch-size", type=int, default=None, help="Override PCA grid batch size.")
    p.add_argument("--input-layer", default=None, help="Input layer to flatten (default: first key).")
    p.add_argument(
        "--output-root",
        default="/home/filip/paper_simulation_results_/cases",
        help="Root folder for outputs (creates a timestamped subfolder).",
    )
    args = p.parse_args(argv)

    outputs = run_pca_flatten(
        config_path=args.config,
        model_key=args.model_key,
        weights_pt=args.weights_pt,
        num_iterations=args.num_iterations,
        pca_steps=args.pca_steps,
        pca_n_sigma=args.pca_n_sigma,
        pca_batch_size=args.pca_batch_size,
        input_layer=args.input_layer,
        output_root=args.output_root,
    )

    print(f"Wrote: {outputs['pca_out']}")
    print(f"Wrote: {outputs['weights_npz']}")
    print(f"Wrote: {outputs['weights_flat']}")
    print(f"Wrote: {outputs['inputs_flat']}")
    print(f"Wrote: {outputs['shapes_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
