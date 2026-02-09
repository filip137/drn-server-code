#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _load_mean_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if isinstance(v, (int, float)) and k != "num_layers"}


def _iter_from_dir(path: Path) -> int:
    match = re.search(r"iter(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse iteration from folder name: {path.name}")
    return int(match.group(1))


def plot_mean_error_summaries(
    base_dir: Path,
    output_dir: Path | None = None,
    metrics: list[str] | None = None,
    log_scale: bool = False,
    log_x: bool = False,
) -> list[Path]:
    base_dir = base_dir.expanduser().resolve()
    output_dir = (output_dir or base_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    candidates = sorted(base_dir.glob("iter*/mean_error_summary.json"))
    if not candidates:
        candidates = sorted(base_dir.rglob("mean_error_summary.json"))
    for path in candidates:
        iter_dir = path.parent
        try:
            iter_val = _iter_from_dir(iter_dir)
        except ValueError:
            continue
        stats = _load_mean_summary(path)
        entries.append((iter_val, stats))

    if not entries:
        raise SystemExit(f"No mean_error_summary.json files found under {base_dir}")

    entries.sort(key=lambda x: x[0])
    iters = [item[0] for item in entries]

    all_keys = sorted({k for _, stats in entries for k in stats.keys()})
    if metrics:
        keys = [k for k in metrics if k in all_keys]
        missing = set(metrics) - set(keys)
        if missing:
            raise SystemExit(f"Requested metrics not found: {sorted(missing)}")
    else:
        keys = all_keys

    series = {
        "iterations": iters,
        "metrics": {},
    }
    outputs = []
    for key in keys:
        values = []
        for _, stats in entries:
            if key not in stats:
                raise SystemExit(f"Missing key '{key}' in one of the summaries.")
            values.append(float(stats[key]))
        series["metrics"][key] = values

        fig, ax = plt.subplots()
        y_vals = values
        if log_scale:
            y_vals = [max(v, 1e-16) for v in y_vals]
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.plot(iters, y_vals, marker="o")
        ax.set_title(key)
        ax.set_xlabel("iterations")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = output_dir / f"mean_error_{key}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        outputs.append(out_path)

    data_path = output_dir / "mean_error_vs_iter.json"
    data_path.write_text(json.dumps(series, indent=2))
    outputs.append(data_path)

    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot mean_error_summary.json metrics vs iteration count.",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing iter*/mean_error_summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots (defaults to base dir).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional subset of metric keys to plot.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale on the y-axis.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use log scale on the x-axis.",
    )
    args = parser.parse_args(argv)

    outputs = plot_mean_error_summaries(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        metrics=args.metrics,
        log_scale=args.log_scale,
        log_x=args.log_x,
    )
    for path in outputs:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
