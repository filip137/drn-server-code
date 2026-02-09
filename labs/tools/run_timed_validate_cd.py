#!/usr/bin/env python3
import argparse
import csv
import json
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


def build_linspace_inputs(
    *,
    linspace_min: float,
    linspace_max: float,
    linspace_samples: int,
) -> np.ndarray:
    xs = np.linspace(linspace_min, linspace_max, linspace_samples)
    ys = np.linspace(linspace_min, linspace_max, linspace_samples)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    return np.hstack([grid, -grid])


def write_config(path: Path, layer_shapes, voltage_amp, current_amp, v_off):
    config = {
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "layer_shapes": layer_shapes,
        "non_linearity": {
            "type": "double_diode_exponential",
            "exponential_params": {
                "V_off": v_off
            }
        },
    }
    path.write_text(json.dumps(config, indent=2))


def force_exponential_config(path: Path, v_off: float) -> None:
    data = json.loads(path.read_text())
    data["non_linearity"] = {
        "type": "double_diode_exponential",
        "exponential_params": {"V_off": v_off},
    }
    data.pop("exponential_diode_param", None)
    path.write_text(json.dumps(data, indent=2))


def write_weights(path: Path, layer_shapes, rng: np.random.Generator):
    arrays = {}
    for idx in range(len(layer_shapes) - 1):
        in_dim = layer_shapes[idx][0]
        out_dim = layer_shapes[idx + 1][0]
        arrays[f"param_{idx}"] = rng.uniform(1e-7, 1e-5, size=(in_dim, out_dim)).astype(np.float64)
    np.savez(path, **arrays)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Time validate_cd_results linspace runs over hidden sizes/layers.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/filip/paper_maxicao_simulations/train_and_validate/timing_validate_cd",
        help="Base output directory for timing results.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[5, 10, 25, 50, 100, 250, 500],
        help="Hidden layer sizes to sweep.",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Hidden layer counts to sweep.",
    )
    parser.add_argument("--linspace-min", type=float, default=-10.0)
    parser.add_argument("--linspace-max", type=float, default=10.0)
    parser.add_argument("--linspace-samples", type=int, default=30)
    parser.add_argument("--voltage-amp", type=float, default=1.0)
    parser.add_argument("--current-amp", type=float, default=1.0)
    parser.add_argument("--v-off", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights-root",
        default=None,
        help="Root folder containing moons_training_sweep h*_d*/<run>/model.npz",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.weights_root:
        output_dir = output_dir / "with_trained_weights_second_run"
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    results = []
    json_path = output_dir / "timing_validate_cd_results.json"
    csv_path = output_dir / "timing_validate_cd_results.csv"
    if json_path.exists():
        try:
            results = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            results = []

    for hidden_layers in args.hidden_layers:
        for hidden_dim in args.hidden_dims:
            layer_shapes = [[4]] + [[hidden_dim]] * hidden_layers + [[4]]
            run_dir = output_dir / f"h{hidden_layers}_d{hidden_dim}"
            run_dir.mkdir(parents=True, exist_ok=True)

            config_path = run_dir / "config.json"
            weights_path = run_dir / "weights.npz"
            inputs_path = run_dir / "linspace_inputs.npz"

            if args.weights_root:
                sweep_root = Path(args.weights_root).expanduser().resolve()
                sweep_dir = sweep_root / f"h{hidden_layers}_d{hidden_dim}"
                if not sweep_dir.exists():
                    raise SystemExit(f"Missing sweep dir: {sweep_dir}")
                runs = [p for p in sweep_dir.iterdir() if p.is_dir()]
                if not runs:
                    raise SystemExit(f"No run directories in {sweep_dir}")
                run_src = sorted(runs)[-1]
                print(f"Using run {run_src.name} for h{hidden_layers}_d{hidden_dim}")
                model_path = run_src / "model.npz"
                config_src = run_src / "run_metadata.json"
                inputs_src = run_src / "linspace_inputs.npz"
                if not model_path.exists():
                    raise SystemExit(f"Missing model.npz in {run_src}")
                if not config_src.exists():
                    raise SystemExit(f"Missing run_metadata.json in {run_src}")
                if not inputs_src.exists():
                    raise SystemExit(f"Missing linspace_inputs.npz in {run_src}")
                weights_path.write_bytes(model_path.read_bytes())
                config_path.write_bytes(config_src.read_bytes())
                force_exponential_config(config_path, args.v_off)
                data = np.load(inputs_src)
                if "Layer_0" in data:
                    X = data["Layer_0"]
                elif "inputs" in data:
                    X = data["inputs"]
                else:
                    raise SystemExit(f"Missing inputs in {inputs_src}")
                if X.shape[1] == 2:
                    X = np.hstack([X, -X])
                np.savez(inputs_path, Layer_0=X)
            else:
                write_config(
                    config_path,
                    layer_shapes,
                    args.voltage_amp,
                    args.current_amp,
                    args.v_off,
                )
                write_weights(weights_path, layer_shapes, rng)

                inputs = build_linspace_inputs(
                    linspace_min=args.linspace_min,
                    linspace_max=args.linspace_max,
                    linspace_samples=args.linspace_samples,
                )
                np.savez(inputs_path, Layer_0=inputs)

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

            result = {
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "linspace_samples": args.linspace_samples,
                "elapsed_seconds": elapsed,
                "run_dir": str(run_dir),
            }
            results.append(result)
            print(
                f"hidden_dim={hidden_dim} hidden_layers={hidden_layers} "
                f"elapsed={elapsed:.4f}s"
            )
            json_path.write_text(json.dumps(results, indent=2))
            if results:
                fieldnames = list(results[0].keys())
                with csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
