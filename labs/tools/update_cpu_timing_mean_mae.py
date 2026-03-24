#!/usr/bin/env python3
"""Add node-weighted mean absolute error to CPU timing summaries.

This updates the generated cpu_timing bundles by computing
`mean_mae_node_weighted` from the per-layer error summaries that sit next to a
`cross_layer_rel_l1_percentiles_node_weighted.json` file.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate mean_mae_node_weighted in cpu_timing/cpu_error_summary.json "
            "and the companion combined_latest_by_hidden.csv."
        )
    )
    parser.add_argument(
        "cpu_timing_dirs",
        nargs="+",
        type=Path,
        help="One or more cpu_timing directories to update.",
    )
    return parser.parse_args()


def compute_mean_mae_node_weighted(error_summary_path: Path) -> Optional[float]:
    if not error_summary_path.exists():
        return None

    payload = json.loads(error_summary_path.read_text())
    node_counts = payload.get("node_counts") or {}
    total_nodes = int(payload.get("total_nodes") or 0)
    if not node_counts or total_nodes <= 0:
        return None

    weighted_sum = 0.0
    for layer, node_count in node_counts.items():
        layer_summary_path = (
            error_summary_path.parent / f"{layer}_vs_{layer}_error_summary.json"
        )
        if not layer_summary_path.exists():
            return None
        layer_payload = json.loads(layer_summary_path.read_text())
        mean_mae = layer_payload.get("mean_mae")
        if mean_mae is None:
            return None
        weighted_sum += float(mean_mae) * int(node_count)

    return weighted_sum / total_nodes


def update_cpu_error_summary(summary_path: Path) -> Dict[Tuple[int, int], Optional[float]]:
    payload = json.loads(summary_path.read_text())
    mean_mae_by_pair: Dict[Tuple[int, int], Optional[float]] = {}

    for row in payload.get("rows", []):
        error_summary_value = row.get("error_summary")
        mean_mae_value: Optional[float] = None
        if error_summary_value:
            mean_mae_value = compute_mean_mae_node_weighted(Path(error_summary_value))
        row["mean_mae_node_weighted"] = mean_mae_value
        mean_mae_by_pair[(int(row["depth"]), int(row["width"]))] = mean_mae_value

    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(payload, indent=2))
    return mean_mae_by_pair


def update_combined_csv(
    csv_path: Path, mean_mae_by_pair: Dict[Tuple[int, int], Optional[float]]
) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    existing_fields = list(rows[0].keys()) if rows else []
    target_field = "mean_mae_node_weighted"

    if target_field in existing_fields:
        fieldnames = existing_fields
    else:
        insert_after = "p99_rel_l1_node_weighted"
        fieldnames = []
        for name in existing_fields:
            fieldnames.append(name)
            if name == insert_after:
                fieldnames.append(target_field)
        if target_field not in fieldnames:
            fieldnames.append(target_field)

    for row in rows:
        pair = (int(row["depth"]), int(row["width"]))
        value = mean_mae_by_pair.get(pair)
        row[target_field] = "" if value is None else repr(value)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_cpu_timing_dir(cpu_timing_dir: Path) -> None:
    summary_path = cpu_timing_dir / "cpu_error_summary.json"
    combined_csv_path = cpu_timing_dir / "combined_latest_by_hidden.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not combined_csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {combined_csv_path}")

    mean_mae_by_pair = update_cpu_error_summary(summary_path)
    update_combined_csv(combined_csv_path, mean_mae_by_pair)


def main() -> None:
    args = parse_args()
    for cpu_timing_dir in args.cpu_timing_dirs:
        process_cpu_timing_dir(cpu_timing_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
