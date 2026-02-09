import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

LABS_DIR = Path(__file__).resolve().parent
MNIST_TRAIN = LABS_DIR / "mnist_train.py"

SUPPORTED_PARAMS = {"beta", "input_gain", "weight_gain", "lr"}


def _resolve_config(path: str) -> Path:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = LABS_DIR / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_path


def _generate_values(
    values: Sequence[float],
    vmin: float,
    vmax: float,
    steps: int,
    scale: str,
) -> List[float]:
    if values:
        return [float(v) for v in values]
    if steps < 1:
        raise ValueError("steps must be >= 1 when values are not provided.")
    if vmin is None or vmax is None:
        raise ValueError("min/max required when explicit values are not provided.")
    if scale == "log":
        if vmin <= 0 or vmax <= 0:
            raise ValueError("Log scale requires positive min/max.")
        arr = np.logspace(math.log10(vmin), math.log10(vmax), steps)
    else:
        arr = np.linspace(vmin, vmax, steps)
    return [float(v) for v in arr]


def _apply_override(config: dict, model_key: str, param: str, value: float) -> None:
    if param == "beta":
        config["beta"] = value
    elif param == "input_gain":
        config["model_base"]["input_gain"] = value
    elif param == "weight_gain":
        overrides = config["model_overrides"].get(model_key)
        if overrides is None:
            raise KeyError(f"Model override '{model_key}' missing from config.")
        gains = overrides.get("weight_gains", [])
        if not gains:
            raise ValueError(f"Model '{model_key}' has no weight_gains list to override.")
        overrides["weight_gains"] = [value for _ in gains]
    elif param == "lr":
        config["lr"] = value
    else:
        raise ValueError(f"Unsupported sweep parameter '{param}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight sweep runner for labs/mnist_train.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.json", help="Base config path.")
    parser.add_argument("--param", required=True, choices=sorted(SUPPORTED_PARAMS), help="Parameter to sweep.")
    parser.add_argument("--values", type=str, default=None, help="Comma-separated explicit values.")
    parser.add_argument("--min", dest="min_value", type=float, default=None, help="Minimum value (when --values unset).")
    parser.add_argument("--max", dest="max_value", type=float, default=None, help="Maximum value (when --values unset).")
    parser.add_argument("--steps", type=int, default=5, help="Number of sweep points (when --values unset).")
    parser.add_argument("--scale", choices=("linear", "log"), default="linear", help="Spacing for generated values.")
    parser.add_argument("--model-key", default="mnist", help="Model override key to edit.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for each run.")
    parser.add_argument("--output", default="sweeps", help="Directory to store sweep artifacts.")
    args = parser.parse_args()

    cfg_path = _resolve_config(args.config)
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    explicit_values = []
    if args.values:
        explicit_values = [float(x.strip()) for x in args.values.split(",") if x.strip()]

    sweep_values = _generate_values(explicit_values, args.min_value, args.max_value, args.steps, args.scale)

    base_config = json.loads(cfg_path.read_text())

    results = []
    for idx, value in enumerate(sweep_values):
        run_dir = output_root / f"{args.param}_{value:.6g}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config_copy = copy.deepcopy(base_config)
        config_copy.setdefault("lab", {})["model_key"] = args.model_key
        _apply_override(config_copy, args.model_key, args.param, value)

        cfg_override = run_dir / "config.override.json"
        cfg_override.write_text(json.dumps(config_copy, indent=2))

        result_path = run_dir / "result.json"
        cmd = [
            sys.executable,
            str(MNIST_TRAIN),
            "--config",
            str(cfg_override),
            "--result-json",
            str(result_path),
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(int(args.epochs))])
        if args.param == "lr":
            cmd.extend(["--lr", f"{value:.8f}"])
        elif args.param == "beta":
            cmd.extend(["--beta", f"{value:.8f}"])

        print(f"[{idx+1}/{len(sweep_values)}] Running {args.param}={value}")
        completed = subprocess.run(cmd, text=True)
        if completed.returncode != 0:
            print(f"Run for value={value} failed (exit {completed.returncode}). See console for details.")
            continue

        payload = json.loads(result_path.read_text())
        results.append({"value": value, **payload})

    if results:
        summary_path = output_root / "sweep_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        best = min(results, key=lambda item: item.get("test_error", float("inf")))
        print(f"Best sweep value: {best['value']} (test_error={best.get('test_error')})")
        print(f"Full summary stored at {summary_path}")
    else:
        print("No successful sweep runs were recorded.")


if __name__ == "__main__":
    main()
