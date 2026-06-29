#!/usr/bin/env python3
"""Select one Conv DRN iteration count per nonlinearity from EP/BP cosine rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "non_linearity",
    "selected_iteration_count",
    "num_amp_rows",
    "amp_selected_iteration_counts",
    "max_reference_cosine_mean",
    "min_selected_cosine_mean",
    "cap",
    "convergence_limited",
    "selection_rule",
]


def _float(value: object, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _select(rows: list[dict], *, cap: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        non_linearity = str(row.get("non_linearity", "")).strip()
        if not non_linearity:
            continue
        grouped.setdefault(non_linearity, []).append(row)

    selected_rows: list[dict] = []
    for non_linearity, group in sorted(grouped.items()):
        selected_ks = [int(_float(row.get("selected_iteration_count"), 0.0)) for row in group]
        selected_ks = [value for value in selected_ks if value > 0]
        if not selected_ks:
            continue
        selected_k = min(max(selected_ks), int(cap))
        highest_ks = [int(_float(row.get("highest_iteration_count"), 0.0)) for row in group]
        convergence_limited = any(
            int(_float(row.get("selected_iteration_count"), 0.0)) >= int(cap)
            and int(_float(row.get("highest_iteration_count"), 0.0)) >= int(cap)
            for row in group
        )
        selected_rows.append(
            {
                "non_linearity": non_linearity,
                "selected_iteration_count": selected_k,
                "num_amp_rows": len(group),
                "amp_selected_iteration_counts": " ".join(str(value) for value in selected_ks),
                "max_reference_cosine_mean": max(
                    _float(row.get("reference_cosine_mean")) for row in group
                ),
                "min_selected_cosine_mean": min(
                    _float(row.get("selected_cosine_mean")) for row in group
                ),
                "cap": int(cap),
                "convergence_limited": bool(
                    convergence_limited or (highest_ks and max(highest_ks) >= int(cap) and selected_k >= int(cap))
                ),
                "selection_rule": (
                    "max selected K across amplification settings, capped at requested cap; "
                    "per-amp K is from EP/BP cosine selected_k_by_amp"
                ),
            }
        )
    return selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosine-selected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(Path(args.cosine_selected).expanduser().resolve())
    selected = _select(rows, cap=int(args.cap))
    output = Path(args.output).expanduser().resolve()
    _write_rows(output, selected)
    print(f"[select-k] rows={len(rows)} selected={len(selected)} output={output}")


if __name__ == "__main__":
    main()
