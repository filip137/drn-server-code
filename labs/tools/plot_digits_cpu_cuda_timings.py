#!/usr/bin/env python3
"""Extract and plot digits validation timings for CPU/CUDA reruns.

This scans a timing tree for `codex*_cpu_rerun` / `codex*_cuda_rerun`
validation reruns, selects the latest rerun per (depth, width, device), writes a
CSV summary, and generates a log-log plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


RERUN_ROOT_RE = re.compile(r"^codex(?:_[A-Za-z0-9]+)*_(cpu|cuda)_rerun$")
TIME_LOG_NAME = "time_digits_validate_with_residual.log"
TIMESTAMP_RE = re.compile(r"^\d{8}-\d{6}")
USER_TIME_RE = re.compile(r"^\s*User time \(seconds\):\s*([0-9]+(?:\.[0-9]+)?)\s*$")
ELAPSED_RE = re.compile(
    r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:. -]+)\s*$"
)


@dataclass(frozen=True)
class Candidate:
    depth: int
    width: int
    device: str
    rerun_timestamp: str
    metadata_path: Path
    log_path: Path
    user_time_seconds: Optional[float]
    wall_time_seconds: Optional[float]
    validation_total_seconds: Optional[float]
    accuracy: Optional[float]
    avg_iterations: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract latest CPU/CUDA digits validation rerun timings and create a "
            "log-log plot while skipping missing runs."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Timing tree root, e.g. .../timings/single_diode_exponential",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for latest CPU/CUDA timings by hidden pair.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        required=True,
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        required=True,
        help="Output plot path (PNG/SVG/PDF based on extension).",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Plot title.",
    )
    return parser.parse_args()


def parse_hidden_pair(path: Path) -> Tuple[int, int]:
    hidden_parts = [part for part in path.parts if re.fullmatch(r"hidden_\d+", part)]
    if len(hidden_parts) < 2:
        raise ValueError(
            f"Expected at least two hidden_* path components, got path={path}"
        )
    return int(hidden_parts[0].split("_", 1)[1]), int(hidden_parts[1].split("_", 1)[1])


def parse_elapsed_to_seconds(value: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped:
        return None

    if "-" in stripped:
        day_part, stripped = stripped.split("-", 1)
        try:
            day_count = int(day_part)
        except ValueError:
            return None
    else:
        day_count = 0

    parts = stripped.split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            total = minutes * 60 + seconds
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total = hours * 3600 + minutes * 60 + seconds
        else:
            return None
    except ValueError:
        return None
    return day_count * 86400 + total


def parse_time_log(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not log_path.exists():
        return None, None

    user_time_seconds: Optional[float] = None
    wall_time_seconds: Optional[float] = None

    for line in log_path.read_text().splitlines():
        match = USER_TIME_RE.match(line)
        if match:
            user_time_seconds = float(match.group(1))
            continue

        match = ELAPSED_RE.match(line)
        if match:
            wall_time_seconds = parse_elapsed_to_seconds(match.group(1))

    return user_time_seconds, wall_time_seconds


def load_candidate(metadata_path: Path, base_path: Path) -> Candidate:
    metadata = json.loads(metadata_path.read_text())
    rerun_dir = metadata_path.parent
    rerun_timestamp = rerun_dir.name
    rerun_root_name = metadata_path.parents[1].name
    rerun_root_match = RERUN_ROOT_RE.fullmatch(rerun_root_name)
    if rerun_root_match is None:
        raise ValueError(
            f"Expected rerun root matching codex*_cpu_rerun/codex*_cuda_rerun; "
            f"got {rerun_root_name}"
        )
    device = rerun_root_match.group(1)
    depth, width = parse_hidden_pair(metadata_path.relative_to(base_path))
    log_path = rerun_dir / TIME_LOG_NAME
    user_time_seconds, wall_time_seconds = parse_time_log(log_path)

    equilibrium_iterations = metadata.get("equilibrium_iterations") or {}
    return Candidate(
        depth=depth,
        width=width,
        device=device,
        rerun_timestamp=rerun_timestamp,
        metadata_path=metadata_path,
        log_path=log_path,
        user_time_seconds=user_time_seconds,
        wall_time_seconds=wall_time_seconds,
        validation_total_seconds=metadata.get("validation_total_seconds"),
        accuracy=metadata.get("validation_accuracy"),
        avg_iterations=equilibrium_iterations.get("avg_iterations"),
    )


def find_latest_candidates(base_path: Path) -> Dict[Tuple[int, int, str], Candidate]:
    selected: Dict[Tuple[int, int, str], Candidate] = {}
    for metadata_path in base_path.glob("**/*_rerun/*/validation_metadata.json"):
        rerun_root_name = metadata_path.parents[1].name
        if RERUN_ROOT_RE.fullmatch(rerun_root_name) is None:
            continue
        candidate = load_candidate(metadata_path, base_path)
        key = (candidate.depth, candidate.width, candidate.device)
        existing = selected.get(key)
        if existing is None or candidate.rerun_timestamp > existing.rerun_timestamp:
            selected[key] = candidate
    return selected


def discover_hidden_pairs(base_path: Path) -> List[Tuple[int, int]]:
    hidden_pairs = set()
    for depth_dir in base_path.glob("hidden_*"):
        if not depth_dir.is_dir():
            continue
        depth_match = re.fullmatch(r"hidden_(\d+)", depth_dir.name)
        if depth_match is None:
            continue
        depth = int(depth_match.group(1))
        for width_dir in depth_dir.glob("hidden_*"):
            if not width_dir.is_dir():
                continue
            width_match = re.fullmatch(r"hidden_(\d+)", width_dir.name)
            if width_match is None:
                continue
            width = int(width_match.group(1))
            hidden_pairs.add((depth, width))
    return sorted(hidden_pairs)


def build_rows(
    all_hidden_pairs: Iterable[Tuple[int, int]],
    selected: Dict[Tuple[int, int, str], Candidate],
    base_path: Path,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    rows: List[dict] = []
    missing_cpu_pairs: List[dict] = []
    missing_cuda_pairs: List[dict] = []
    missing_both_pairs: List[dict] = []

    for depth, width in all_hidden_pairs:
        cpu = selected.get((depth, width, "cpu"))
        cuda = selected.get((depth, width, "cuda"))

        if cpu is None:
            missing_cpu_pairs.append({"depth": depth, "width": width})
        if cuda is None:
            missing_cuda_pairs.append({"depth": depth, "width": width})
        if cpu is None and cuda is None:
            missing_both_pairs.append({"depth": depth, "width": width})

        row = {
            "depth": depth,
            "width": width,
            "cpu_user_time_seconds": cpu.user_time_seconds if cpu else None,
            "cpu_wall_time_seconds": cpu.wall_time_seconds if cpu else None,
            "cpu_validation_total_seconds": (
                cpu.validation_total_seconds if cpu else None
            ),
            "cpu_accuracy": cpu.accuracy if cpu else None,
            "cpu_avg_iterations": cpu.avg_iterations if cpu else None,
            "cpu_rerun_timestamp": cpu.rerun_timestamp if cpu else "",
            "cpu_metadata_rel_path": (
                str(cpu.metadata_path.relative_to(base_path)) if cpu else ""
            ),
            "cpu_log_rel_path": str(cpu.log_path.relative_to(base_path)) if cpu else "",
            "cuda_user_time_seconds": cuda.user_time_seconds if cuda else None,
            "cuda_wall_time_seconds": cuda.wall_time_seconds if cuda else None,
            "cuda_validation_total_seconds": (
                cuda.validation_total_seconds if cuda else None
            ),
            "cuda_accuracy": cuda.accuracy if cuda else None,
            "cuda_avg_iterations": cuda.avg_iterations if cuda else None,
            "cuda_rerun_timestamp": cuda.rerun_timestamp if cuda else "",
            "cuda_metadata_rel_path": (
                str(cuda.metadata_path.relative_to(base_path)) if cuda else ""
            ),
            "cuda_log_rel_path": (
                str(cuda.log_path.relative_to(base_path)) if cuda else ""
            ),
        }
        rows.append(row)

    return rows, missing_cpu_pairs, missing_cuda_pairs, missing_both_pairs


def write_csv(rows: Iterable[dict], output_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No timing rows found to write.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _valid_points(rows: Iterable[dict], device: str, depth: int) -> List[Tuple[int, float]]:
    column = f"{device}_user_time_seconds"
    points: List[Tuple[int, float]] = []
    for row in rows:
        if int(row["depth"]) != depth:
            continue
        value = row[column]
        if value in (None, ""):
            continue
        numeric = float(value)
        if numeric <= 0:
            continue
        points.append((int(row["width"]), numeric))
    return sorted(points, key=lambda item: item[0])


def plot_rows(rows: List[dict], output_plot: Path, title: str) -> None:
    depths = sorted({int(row["depth"]) for row in rows})
    colors = plt.get_cmap("tab10")
    device_styles = {
        "cpu": {"linestyle": "-", "marker": "o"},
        "cuda": {"linestyle": "--", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    plotted_any = False

    for idx, depth in enumerate(depths):
        color = colors(idx % 10)
        for device in ("cpu", "cuda"):
            points = _valid_points(rows, device, depth)
            if not points:
                continue
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linewidth=2.0,
                markersize=6,
                label=f"{depth}H {device.upper()}",
                **device_styles[device],
            )
            plotted_any = True

    if not plotted_any:
        raise ValueError("No valid CPU/CUDA timing points were found to plot.")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Hidden size")
    ax.set_ylabel("User time (seconds)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=220)
    plt.close(fig)


def write_summary(
    *,
    base_path: Path,
    output_csv: Path,
    output_summary: Path,
    output_plot: Path,
    rows: List[dict],
    missing_cpu_pairs: List[dict],
    missing_cuda_pairs: List[dict],
    missing_both_pairs: List[dict],
) -> None:
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_path": str(base_path),
        "selection_rule": (
            "latest codex*_cpu_rerun/codex*_cuda_rerun validation_metadata.json "
            "per (depth,width,device), ordered by rerun timestamp directory name"
        ),
        "counts": {
            "hidden_pairs_total": len(rows),
            "hidden_pairs_with_any_device": sum(
                1
                for row in rows
                if row["cpu_metadata_rel_path"] or row["cuda_metadata_rel_path"]
            ),
            "cpu_pairs_selected": sum(1 for row in rows if row["cpu_metadata_rel_path"]),
            "cuda_pairs_selected": sum(
                1 for row in rows if row["cuda_metadata_rel_path"]
            ),
        },
        "output_files": {
            "csv": str(output_csv),
            "plot": str(output_plot),
            "summary": str(output_summary),
        },
        "missing_cpu_pairs": missing_cpu_pairs,
        "missing_cuda_pairs": missing_cuda_pairs,
        "missing_both_pairs": missing_both_pairs,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()
    all_hidden_pairs = discover_hidden_pairs(base_path)
    if not all_hidden_pairs:
        raise ValueError(f"No hidden depth/width folders found under {base_path}")
    selected = find_latest_candidates(base_path)
    if not selected:
        raise ValueError(
            f"No matching codex*_cpu_rerun/codex*_cuda_rerun metadata found under {base_path}"
        )

    rows, missing_cpu_pairs, missing_cuda_pairs, missing_both_pairs = build_rows(
        all_hidden_pairs, selected, base_path
    )
    output_csv = args.output_csv.expanduser().resolve()
    output_summary = args.output_summary.expanduser().resolve()
    output_plot = args.output_plot.expanduser().resolve()

    write_csv(rows, output_csv)
    plot_rows(rows, output_plot, args.title)
    write_summary(
        base_path=base_path,
        output_csv=output_csv,
        output_summary=output_summary,
        output_plot=output_plot,
        rows=rows,
        missing_cpu_pairs=missing_cpu_pairs,
        missing_cuda_pairs=missing_cuda_pairs,
        missing_both_pairs=missing_both_pairs,
    )

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote plot: {output_plot}")
    print(f"Wrote summary: {output_summary}")
    print(f"Missing CPU pairs: {len(missing_cpu_pairs)}")
    print(f"Missing CUDA pairs: {len(missing_cuda_pairs)}")
    print(f"Missing both-device pairs: {len(missing_both_pairs)}")


if __name__ == "__main__":
    main()
