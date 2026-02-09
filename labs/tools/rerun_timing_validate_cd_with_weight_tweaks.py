#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/filip")
SIM_DIR = PROJECT_ROOT / "simulations" / "improved_simulation_functions"
if str(SIM_DIR) not in sys.path:
    sys.path.append(str(SIM_DIR))

from simulation_parameters_folder import SimulationParametersDCEvaluator  # noqa: E402
import validate_cd_results  # noqa: E402


def tweak_weights(path: Path) -> None:
    data = np.load(path)
    arrays = {}
    for key in data.files:
        arr = data[key].copy()
        flat = arr.reshape(-1)
        # Set every 10th entry to 10, then every 5th to 1e-7 (10th wins last).
        flat[9::10] = 10.0
        flat[4::5] = 1e-7
        arrays[key] = flat.reshape(arr.shape)
    np.savez(path, **arrays)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Tweak weights and rerun validate_cd_results for each timing run.",
    )
    parser.add_argument(
        "--runs-dir",
        default="/home/filip/paper_maxicao_simulations/train_and_validate/timing_validate_cd",
        help="Directory containing timing run subfolders.",
    )
    parser.add_argument("--time", action="store_true", help="Print elapsed time per run.")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("h")])
    if not run_dirs:
        raise SystemExit(f"No run dirs found in {runs_dir}")

    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        weights_path = run_dir / "weights.npz"
        inputs_path = run_dir / "linspace_inputs.npz"

        if not (config_path.exists() and weights_path.exists() and inputs_path.exists()):
            print(f"Skipping {run_dir} (missing config/weights/inputs)")
            continue

        tweak_weights(weights_path)
        sim_params = SimulationParametersDCEvaluator(str(config_path))

        start = time.perf_counter()
        validate_cd_results.main(
            sim_params,
            str(inputs_path),
            str(weights_path),
            config_path=str(config_path),
            validate_mnist=False,
            validate_moons=False,
        )
        elapsed = time.perf_counter() - start
        if args.time:
            print(f"{run_dir.name}: {elapsed:.4f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
