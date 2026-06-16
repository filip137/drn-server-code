#!/usr/bin/env python3
"""Summarize recently completed experiment results.

This is intended for the daily Notion monitor used by the MNIST BP
amplification experiments, but it only assumes the common artifact layout:

    results/<experiment>/<run_name>/seed_<seed>/metrics.json

The script filters completed runs by the modification time of metrics.json.
Recently touched seed directories that do not yet have metrics.json are
reported separately as incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_REPORT_DIR = DEFAULT_RESULTS_ROOT / "daily_result_reports"


@dataclass(frozen=True)
class ResultRow:
    experiment: str
    run_name: str
    seed: str
    voltage_amp: object
    current_amp: object
    training_algorithm: str
    best_test_accuracy: object
    final_test_accuracy: object
    best_epoch: object
    final_train_loss: object
    final_test_loss: object
    checkpoint_path: str
    weights_best_path: str
    weights_final_path: str
    metrics_path: str
    completed_at: str


CSV_COLUMNS = [
    "experiment",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "training_algorithm",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
    "metrics_path",
    "completed_at",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        action="append",
        type=Path,
        default=None,
        help="Results root to scan. Can be passed multiple times. Defaults to ./results.",
    )
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument(
        "--now",
        help="Override current time for tests, as an ISO timestamp in --timezone.",
    )
    return parser.parse_args()


def _now(args: argparse.Namespace) -> datetime:
    tz = ZoneInfo(args.timezone)
    if args.now:
        parsed = datetime.fromisoformat(args.now)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz)
    return datetime.now(tz)


def _mtime(path: Path, tz: ZoneInfo) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _seed_from_path(metrics_path: Path, metrics: dict) -> str:
    if "seed" in metrics:
        return str(metrics["seed"])
    seed_dir = metrics_path.parent.name
    if seed_dir.startswith("seed_"):
        return seed_dir.removeprefix("seed_")
    return seed_dir


def _experiment_from_path(root: Path, metrics_path: Path) -> str:
    try:
        rel = metrics_path.relative_to(root)
    except ValueError:
        return metrics_path.parents[2].name
    return rel.parts[0] if len(rel.parts) >= 4 else root.name


def _run_from_path(metrics_path: Path) -> str:
    return metrics_path.parents[1].name


def _result_row(root: Path, metrics_path: Path, tz: ZoneInfo) -> ResultRow:
    metrics = _load_json(metrics_path)
    completed_at = _mtime(metrics_path, tz).isoformat(timespec="seconds")
    return ResultRow(
        experiment=_experiment_from_path(root, metrics_path),
        run_name=_run_from_path(metrics_path),
        seed=_seed_from_path(metrics_path, metrics),
        voltage_amp=metrics.get("voltage_amp", ""),
        current_amp=metrics.get("current_amp", ""),
        training_algorithm=str(metrics.get("training_algorithm", "")),
        best_test_accuracy=metrics.get("best_test_accuracy", ""),
        final_test_accuracy=metrics.get("final_test_accuracy", ""),
        best_epoch=metrics.get("best_epoch", ""),
        final_train_loss=metrics.get("final_train_loss", ""),
        final_test_loss=metrics.get("final_test_loss", ""),
        checkpoint_path=str(metrics.get("best_checkpoint_path") or metrics.get("checkpoint_path") or ""),
        weights_best_path=str(metrics.get("weights_best_path", "")),
        weights_final_path=str(metrics.get("weights_final_path", "")),
        metrics_path=str(metrics_path),
        completed_at=completed_at,
    )


def _find_recent_completed(
    roots: list[Path],
    cutoff: datetime,
    tz: ZoneInfo,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in root.rglob("metrics.json"):
            resolved = metrics_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if _mtime(metrics_path, tz) < cutoff:
                continue
            rows.append(_result_row(root, metrics_path, tz))
    return sorted(rows, key=lambda row: (row.experiment, row.run_name, int(row.seed) if row.seed.isdigit() else row.seed))


def _find_recent_incomplete(roots: list[Path], cutoff: datetime, tz: ZoneInfo) -> dict[str, int]:
    incomplete: dict[str, set[Path]] = defaultdict(set)
    for root in roots:
        if not root.exists():
            continue
        for config_path in root.rglob("config.json"):
            seed_dir = config_path.parent
            if not seed_dir.name.startswith("seed_"):
                continue
            if (seed_dir / "metrics.json").exists():
                continue
            files = [path for path in seed_dir.iterdir() if path.is_file()]
            if not files:
                continue
            touched = max(_mtime(path, tz) for path in files)
            if touched < cutoff:
                continue
            incomplete[_experiment_from_path(root, seed_dir / "metrics.json")].add(seed_dir)
    return {experiment: len(paths) for experiment, paths in sorted(incomplete.items())}


def _as_float(value: object) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{100.0 * number:.2f}%"


def _num(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    return f"{number:.6g}"


def _aggregate(rows: list[ResultRow]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[ResultRow]] = defaultdict(list)
    for row in rows:
        groups[(row.experiment, row.run_name)].append(row)

    aggregates = []
    for (experiment, run_name), group in sorted(groups.items()):
        finals = [value for value in (_as_float(row.final_test_accuracy) for row in group) if value is not None]
        bests = [value for value in (_as_float(row.best_test_accuracy) for row in group) if value is not None]
        best_row = max(group, key=lambda row: _as_float(row.best_test_accuracy) or float("-inf"))
        aggregates.append(
            {
                "experiment": experiment,
                "run_name": run_name,
                "seeds": ", ".join(row.seed for row in group),
                "n": len(group),
                "best_seed": best_row.seed,
                "best_test_accuracy": best_row.best_test_accuracy,
                "mean_best_test_accuracy": sum(bests) / len(bests) if bests else "",
                "mean_final_test_accuracy": sum(finals) / len(finals) if finals else "",
            }
        )
    return aggregates


def _markdown_report(
    rows: list[ResultRow],
    incomplete: dict[str, int],
    started_at: datetime,
    cutoff: datetime,
    roots: list[Path],
) -> str:
    lines = [
        f"## Daily Experiment Results - {started_at.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"Window: {cutoff.strftime('%Y-%m-%d %H:%M %Z')} to {started_at.strftime('%Y-%m-%d %H:%M %Z')}.",
        f"Scanned: {', '.join(str(root) for root in roots)}.",
        "",
        f"Completed runs found: {len(rows)}.",
    ]

    if incomplete:
        incomplete_text = ", ".join(f"{experiment}: {count}" for experiment, count in incomplete.items())
        lines.append(f"Recently touched incomplete seed dirs: {incomplete_text}.")

    if not rows:
        lines.extend(["", "No completed `metrics.json` files were modified in this window."])
        return "\n".join(lines) + "\n"

    lines.extend(["", "### Aggregate"])
    for aggregate in _aggregate(rows):
        lines.append(
            "- "
            f"{aggregate['experiment']} / {aggregate['run_name']}: "
            f"{aggregate['n']} seed(s) [{aggregate['seeds']}], "
            f"mean best {_pct(aggregate['mean_best_test_accuracy'])}, "
            f"mean final {_pct(aggregate['mean_final_test_accuracy'])}, "
            f"best seed {aggregate['best_seed']} at {_pct(aggregate['best_test_accuracy'])}."
        )

    lines.extend(["", "### Completed Runs"])
    for row in rows:
        lines.append(
            "- "
            f"{row.experiment} / {row.run_name} / seed {row.seed}: "
            f"v_amp={_num(row.voltage_amp)}, c_amp={_num(row.current_amp)}, "
            f"best={_pct(row.best_test_accuracy)} at epoch {row.best_epoch}, "
            f"final={_pct(row.final_test_accuracy)}, "
            f"final train loss={_num(row.final_train_loss)}, "
            f"final test loss={_num(row.final_test_loss)}."
        )
        lines.append(f"  Best weights: `{row.weights_best_path}`")
        lines.append(f"  Final weights: `{row.weights_final_path}`")

    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in CSV_COLUMNS})


def main() -> int:
    args = _parse_args()
    tz = ZoneInfo(args.timezone)
    started_at = _now(args)
    cutoff = started_at - timedelta(hours=args.since_hours)
    roots = [path.resolve() for path in (args.results_root or [DEFAULT_RESULTS_ROOT])]

    rows = _find_recent_completed(roots, cutoff, tz)
    incomplete = _find_recent_incomplete(roots, cutoff, tz)
    markdown = _markdown_report(rows, incomplete, started_at, cutoff, roots)

    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    report_dir = args.report_dir.resolve()
    markdown_output = args.markdown_output or report_dir / f"recent_results_{timestamp}.md"
    csv_output = args.csv_output or report_dir / f"recent_results_{timestamp}.csv"

    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown)
    _write_csv(csv_output, rows)
    (report_dir / "latest.md").write_text(markdown)
    (report_dir / "latest.csv").write_text(csv_output.read_text())

    print(markdown, end="")
    print(f"\nMarkdown report: {markdown_output}")
    print(f"CSV report: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
