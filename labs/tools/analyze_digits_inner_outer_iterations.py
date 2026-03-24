#!/usr/bin/env python3
"""Analyze outer equilibrium iterations and inner Newton iterations for digits runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


@dataclass
class IterationRow:
    family: str
    depth: int
    width: int
    validation_accuracy: float | None
    cpu_user_time_seconds: float | None
    cpu_validation_total_seconds: float | None
    outer_calls: int | None
    outer_avg_iterations: float | None
    outer_max_iterations: int | None
    total_outer_iterations: float | None
    inner_calls: int | None
    inner_total_iterations: int | None
    inner_avg_iterations_per_call: float | None
    inner_calls_per_sample: float | None
    inner_iterations_per_sample: float | None
    inner_calls_per_outer_iteration: float | None
    inner_iterations_per_outer_iteration: float | None
    metadata_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze inner/outer iteration counts for selected CPU timing runs."
        )
    )
    parser.add_argument(
        "--cpu-timing-dir",
        action="append",
        type=Path,
        required=True,
        help="Path to a cpu_timing directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis outputs will be written.",
    )
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_rel_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_family_rows(cpu_timing_dir: Path) -> List[IterationRow]:
    family = cpu_timing_dir.parent.name
    timing_base = cpu_timing_dir.parent
    csv_path = cpu_timing_dir / "combined_latest_by_hidden.csv"
    rows: List[IterationRow] = []

    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            rel_meta = row.get("coord_metadata_rel_path") or ""
            if not rel_meta:
                continue
            metadata_path = timing_base / rel_meta
            metadata = json.loads(metadata_path.read_text())

            outer = metadata.get("equilibrium_iterations") or {}
            inner = metadata.get("newton_iteration_stats") or {}
            outer_calls = outer.get("calls")
            outer_avg = outer.get("avg_iterations")
            outer_max = outer.get("max_iterations")

            total_outer = None
            if outer_calls is not None and outer_avg is not None:
                total_outer = float(outer_calls) * float(outer_avg)

            inner_calls = inner.get("calls")
            inner_total = inner.get("total_iters")

            inner_calls_per_sample = None
            inner_total_per_sample = None
            inner_calls_per_outer = None
            inner_total_per_outer = None
            if outer_calls:
                inner_calls_per_sample = float(inner_calls or 0) / float(outer_calls)
                inner_total_per_sample = float(inner_total or 0) / float(outer_calls)
            if total_outer and total_outer > 0:
                inner_calls_per_outer = float(inner_calls or 0) / total_outer
                inner_total_per_outer = float(inner_total or 0) / total_outer

            rows.append(
                IterationRow(
                    family=family,
                    depth=int(row["depth"]),
                    width=int(row["width"]),
                    validation_accuracy=_to_float(row.get("coord_accuracy")),
                    cpu_user_time_seconds=_to_float(row.get("coord_user_time_seconds")),
                    cpu_validation_total_seconds=_to_float(
                        row.get("coord_validation_total_seconds")
                    ),
                    outer_calls=outer_calls,
                    outer_avg_iterations=outer_avg,
                    outer_max_iterations=outer_max,
                    total_outer_iterations=total_outer,
                    inner_calls=inner_calls,
                    inner_total_iterations=inner_total,
                    inner_avg_iterations_per_call=inner.get("avg_iters_per_call"),
                    inner_calls_per_sample=inner_calls_per_sample,
                    inner_iterations_per_sample=inner_total_per_sample,
                    inner_calls_per_outer_iteration=inner_calls_per_outer,
                    inner_iterations_per_outer_iteration=inner_total_per_outer,
                    metadata_path=str(metadata_path),
                )
            )

    return sorted(rows, key=lambda item: (item.depth, item.width))


def write_csv(rows: Iterable[IterationRow], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_json(rows: Iterable[IterationRow], path: Path, cpu_timing_dir: Path) -> None:
    rows = list(rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cpu_timing_dir": str(cpu_timing_dir),
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2))


def _plot_metric(ax, rows: List[IterationRow], metric: str, title: str, ylabel: str) -> None:
    depths = sorted({row.depth for row in rows})
    colors = plt.get_cmap("tab10")
    for idx, depth in enumerate(depths):
        pts = [(row.width, getattr(row, metric)) for row in rows if row.depth == depth]
        pts = [(w, v) for w, v in pts if v is not None]
        if not pts:
            continue
        pts.sort()
        ax.plot(
            [w for w, _ in pts],
            [v for _, v in pts],
            marker="o",
            linewidth=2.0,
            color=colors(idx % 10),
            label=f"{depth}H",
        )
    ax.set_xscale("log", base=2)
    ax.set_title(title)
    ax.set_xlabel("Hidden size")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.45)


def plot_family(rows: List[IterationRow], family: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    _plot_metric(
        axes[0],
        rows,
        "outer_avg_iterations",
        "Outer Iterations Per Sample",
        "avg outer iterations",
    )
    _plot_metric(
        axes[1],
        rows,
        "inner_iterations_per_sample",
        "Inner Newton Iterations Per Sample",
        "avg inner iterations",
    )
    _plot_metric(
        axes[2],
        rows,
        "inner_iterations_per_outer_iteration",
        "Inner Iterations Per Outer Iteration",
        "inner / outer",
    )
    axes[2].legend(loc="best", fontsize=9)
    fig.suptitle(f"{family}: Inner vs Outer Iteration Analysis")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_family_summary(rows: List[IterationRow]) -> List[str]:
    def fmt(row: IterationRow) -> str:
        return (
            f"depth {row.depth}, width {row.width}: outer={row.outer_avg_iterations:.3f}, "
            f"inner/sample={row.inner_iterations_per_sample:.3f}, "
            f"inner/outer={row.inner_iterations_per_outer_iteration:.3f}"
        )

    nonzero_inner = [r for r in rows if (r.inner_total_iterations or 0) > 0]
    max_outer = max(rows, key=lambda r: r.outer_avg_iterations or float("-inf"))
    max_inner_per_outer = max(
        nonzero_inner,
        key=lambda r: r.inner_iterations_per_outer_iteration or float("-inf"),
    )
    max_inner_per_sample = max(
        nonzero_inner, key=lambda r: r.inner_iterations_per_sample or float("-inf")
    )
    min_inner_per_outer = min(
        nonzero_inner,
        key=lambda r: r.inner_iterations_per_outer_iteration or float("inf"),
    )
    zero_inner = [r for r in rows if (r.inner_total_iterations or 0) == 0]

    lines = [
        f"- highest outer-iteration load: {fmt(max_outer)}",
        f"- highest inner-per-outer load: {fmt(max_inner_per_outer)}",
        f"- highest total inner work per sample: {fmt(max_inner_per_sample)}",
        f"- lowest nonzero inner-per-outer load: {fmt(min_inner_per_outer)}",
    ]
    if zero_inner:
        joined = ", ".join(f"{r.depth}H/{r.width}" for r in zero_inner)
        lines.append(f"- zero-inner-Newton cases: {joined}")
    return lines


def write_markdown(
    family_to_rows: dict[str, List[IterationRow]],
    output_path: Path,
    output_dir: Path,
) -> None:
    lines: List[str] = [
        "# Inner / Outer Iteration Analysis",
        "",
        f"Generated at `{datetime.now().isoformat(timespec='seconds')}`.",
        "",
        "Definitions:",
        "- outer iterations = `equilibrium_iterations.avg_iterations`",
        "- inner iterations = Newton iterations from `newton_iteration_stats`",
        "- `inner_iterations_per_outer_iteration = total_inner_iterations / total_outer_iterations`",
        "",
    ]

    for family, rows in family_to_rows.items():
        lines.append(f"## {family}")
        lines.append("")
        lines.extend(build_family_summary(rows))
        lines.append("")
        lines.append(
            f"- CSV: `{_to_rel_path(output_dir / f'{family}_inner_outer_metrics.csv', output_dir)}`"
        )
        lines.append(
            f"- Plot: `{_to_rel_path(output_dir / f'{family}_inner_outer_iterations.png', output_dir)}`"
        )
        lines.append("")
        lines.append("| depth | width | outer_avg_iterations | inner_avg_iterations_per_call | inner_iterations_per_sample | inner_iterations_per_outer_iteration |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row.depth} | {row.width} | "
                f"{'' if row.outer_avg_iterations is None else f'{row.outer_avg_iterations:.6f}'} | "
                f"{'' if row.inner_avg_iterations_per_call is None else f'{row.inner_avg_iterations_per_call:.6f}'} | "
                f"{'' if row.inner_iterations_per_sample is None else f'{row.inner_iterations_per_sample:.6f}'} | "
                f"{'' if row.inner_iterations_per_outer_iteration is None else f'{row.inner_iterations_per_outer_iteration:.6f}'} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    family_to_rows: dict[str, List[IterationRow]] = {}
    for cpu_timing_dir in args.cpu_timing_dir:
        cpu_timing_dir = cpu_timing_dir.expanduser().resolve()
        rows = load_family_rows(cpu_timing_dir)
        family = cpu_timing_dir.parent.name
        family_to_rows[family] = rows

        write_csv(rows, output_dir / f"{family}_inner_outer_metrics.csv")
        write_json(rows, output_dir / f"{family}_inner_outer_metrics.json", cpu_timing_dir)
        plot_family(rows, family, output_dir / f"{family}_inner_outer_iterations.png")

    write_markdown(family_to_rows, output_dir / "summary.md", output_dir)


if __name__ == "__main__":
    main()
