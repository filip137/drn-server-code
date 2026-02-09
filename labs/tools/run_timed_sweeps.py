#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

FUNCTIONS_DIR = Path(__file__).resolve().parent
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.append(str(FUNCTIONS_DIR))

import time_linspace  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run timed linspace sweeps over hidden sizes and layer counts.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/filip/paper_simulation_results_/timing_runs",
        help="Base output directory for timing results.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[5, 10, 25, 50, 100],
        help="Hidden layer sizes to sweep.",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Hidden layer counts to sweep.",
    )
    parser.add_argument(
        "--linspace-samples",
        type=int,
        default=30,
        help="Linspace samples per axis.",
    )
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu or cuda).")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args(argv)

    iter_map = {1: 16, 2: 64, 3: 128}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for layers in args.hidden_layers:
        if layers not in iter_map:
            raise SystemExit(f"Unsupported hidden layer count: {layers}")
        iters = iter_map[layers]
        for dim in args.hidden_dims:
            result = time_linspace.run_timed_linspace(
                hidden_dim=dim,
                hidden_layers=layers,
                num_iterations=iters,
                linspace_samples=args.linspace_samples,
                output_dir=output_dir,
                device=args.device,
                seed=args.seed,
            )
            results.append(result)
            print(
                f"hidden_dim={dim} hidden_layers={layers} iters={iters} "
                f"elapsed={result['elapsed_seconds']:.4f}s"
            )

    json_path = output_dir / "timing_sweep_results.json"
    json_path.write_text(json.dumps(results, indent=2))

    csv_path = output_dir / "timing_sweep_results.csv"
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
