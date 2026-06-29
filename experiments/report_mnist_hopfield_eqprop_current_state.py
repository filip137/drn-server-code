#!/usr/bin/env python3
"""Summarize Hopfield EqProp Conv MNIST results into current_state.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
START = "<!-- hopfield-eqprop-autoupdate:start -->"
END = "<!-- hopfield-eqprop-autoupdate:end -->"


FINAL_COLUMNS = [
    "conv_depth",
    "seed",
    "lr_multiplier",
    "best_test_accuracy",
    "final_test_accuracy",
    "final_train_accuracy",
    "best_epoch",
    "run_dir",
    "metrics_path",
]
SAT_COLUMNS = [
    "conv_depth",
    "checkpoint_kind",
    "split",
    "seed_count",
    "accuracy_mean",
    "all_total_saturation_mean",
    "sample_total_saturation_mean_mean",
    "sample_total_saturation_p50_mean",
    "sample_total_saturation_p90_mean",
    "layer_total_saturation_mean",
]


def _float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value: object) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{100.0 * number:.2f}%"


def _fmt_float(value: object) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.4g}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _discover_metrics(root: Path) -> list[Path]:
    return sorted(root.expanduser().resolve().glob("**/metrics.json"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.mean(clean) if clean else math.nan


def _std(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0 if clean else math.nan


def _parse_layer_values(value: object) -> list[float]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [_float(item) for item in parsed]


def _mean_layer_values(rows: list[dict]) -> list[float]:
    values = [_parse_layer_values(row.get("layer_total_saturation")) for row in rows]
    width = max((len(item) for item in values), default=0)
    means = []
    for index in range(width):
        means.append(_mean([item[index] for item in values if index < len(item)]))
    return means


def _final_rows(final_root: Path) -> list[dict]:
    rows = []
    for metrics_path in _discover_metrics(final_root):
        metrics = _load_json(metrics_path)
        if metrics.get("run_group") != "final":
            continue
        rows.append(
            {
                "conv_depth": int(metrics.get("conv_depth", -1)),
                "seed": int(metrics.get("seed", -1)),
                "lr_multiplier": _float(metrics.get("lr_multiplier")),
                "best_test_accuracy": _float(metrics.get("best_test_accuracy")),
                "final_test_accuracy": _float(metrics.get("final_test_accuracy")),
                "final_train_accuracy": _float(metrics.get("final_train_accuracy")),
                "best_epoch": metrics.get("best_epoch", ""),
                "run_dir": metrics.get("run_dir", str(metrics_path.parent)),
                "metrics_path": str(metrics_path),
            }
        )
    return sorted(rows, key=lambda row: (row["conv_depth"], row["seed"]))


def _aggregate_final(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["conv_depth"]), []).append(row)
    aggregate = []
    for depth, group in sorted(grouped.items()):
        lr_values = sorted({_float(row["lr_multiplier"]) for row in group})
        aggregate.append(
            {
                "conv_depth": depth,
                "seed_count": len(group),
                "seeds": ",".join(str(int(row["seed"])) for row in sorted(group, key=lambda item: item["seed"])),
                "lr_multiplier": lr_values[0] if len(lr_values) == 1 else math.nan,
                "best_test_accuracy_mean": _mean([row["best_test_accuracy"] for row in group]),
                "best_test_accuracy_std": _std([row["best_test_accuracy"] for row in group]),
                "final_test_accuracy_mean": _mean([row["final_test_accuracy"] for row in group]),
                "final_test_accuracy_std": _std([row["final_test_accuracy"] for row in group]),
            }
        )
    return aggregate


def _aggregate_saturation(saturation_csv: Path) -> list[dict]:
    if not saturation_csv.exists():
        return []
    rows = _read_csv(saturation_csv)
    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for row in rows:
        key = (int(row["conv_depth"]), row["checkpoint_kind"], row["split"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (depth, checkpoint, split), group in sorted(grouped.items()):
        aggregate.append(
            {
                "conv_depth": depth,
                "checkpoint_kind": checkpoint,
                "split": split,
                "seed_count": len({row["seed"] for row in group}),
                "accuracy_mean": _mean([_float(row.get("accuracy")) for row in group]),
                "all_total_saturation_mean": _mean([_float(row.get("all_total_saturation")) for row in group]),
                "sample_total_saturation_mean_mean": _mean(
                    [_float(row.get("sample_total_saturation_mean")) for row in group]
                ),
                "sample_total_saturation_p50_mean": _mean(
                    [_float(row.get("sample_total_saturation_p50")) for row in group]
                ),
                "sample_total_saturation_p90_mean": _mean(
                    [_float(row.get("sample_total_saturation_p90")) for row in group]
                ),
                "layer_total_saturation_mean": json.dumps(_mean_layer_values(group)),
            }
        )
    return aggregate


def _layer_summary(value: object) -> str:
    layers = _parse_layer_values(value)
    if not layers:
        return "n/a"
    return ", ".join(f"h{index + 1}={_pct(layer)}" for index, layer in enumerate(layers))


def _markdown(args: argparse.Namespace, final_rows: list[dict], final_agg: list[dict], sat_agg: list[dict]) -> str:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        START,
        f"- Updated by Hopfield EqProp report job at {updated}.",
        f"- Result root: `{args.output_root}`.",
        f"- Jobs: LR screen `{args.lr_job}`, LR collector `{args.collect_job}`, final training `{args.final_job}`, saturation `{args.saturation_job}`.",
        f"- Final training summary CSV: `{args.final_summary_csv}`.",
        f"- Saturation CSV: `{args.saturation_csv}`.",
        f"- Saturation summary CSV: `{args.saturation_summary_csv}`.",
        "",
    ]
    if final_agg:
        lines.extend(
            [
                "Final training mean over seeds:",
                "",
                "| Conv | Seeds | LR mult | Best test acc mean | Final test acc mean |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in final_agg:
            lines.append(
                "| Conv{conv_depth} | {seeds} | {lr} | {best} | {final} |".format(
                    conv_depth=row["conv_depth"],
                    seeds=row["seeds"],
                    lr=_fmt_float(row["lr_multiplier"]),
                    best=f"{_pct(row['best_test_accuracy_mean'])} +/- {_pct(row['best_test_accuracy_std'])}",
                    final=f"{_pct(row['final_test_accuracy_mean'])} +/- {_pct(row['final_test_accuracy_std'])}",
                )
            )
        lines.append("")
    elif final_rows:
        lines.append("- Final metrics were found, but no depth aggregate could be built.")
        lines.append("")
    else:
        lines.append("- Final metrics are not available yet.")
        lines.append("")

    final_test = [
        row
        for row in sat_agg
        if row["checkpoint_kind"] == "final" and row["split"] == "test"
    ]
    if final_test:
        lines.extend(
            [
                "Final-checkpoint saturation on the 2048-sample test subset:",
                "",
                "| Conv | Accuracy | All sat | Sample sat mean | Sample sat p50 | Sample sat p90 | Layer totals |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in final_test:
            lines.append(
                "| Conv{conv_depth} | {accuracy} | {all_sat} | {sample_mean} | {sample_p50} | {sample_p90} | {layers} |".format(
                    conv_depth=row["conv_depth"],
                    accuracy=_pct(row["accuracy_mean"]),
                    all_sat=_pct(row["all_total_saturation_mean"]),
                    sample_mean=_pct(row["sample_total_saturation_mean_mean"]),
                    sample_p50=_pct(row["sample_total_saturation_p50_mean"]),
                    sample_p90=_pct(row["sample_total_saturation_p90_mean"]),
                    layers=_layer_summary(row["layer_total_saturation_mean"]),
                )
            )
        lines.append("")

    checkpoint_rows = [
        row
        for row in sat_agg
        if row["split"] == "test" and row["checkpoint_kind"] in {"init", "best", "final"}
    ]
    if checkpoint_rows:
        by_depth: dict[int, dict[str, dict]] = {}
        for row in checkpoint_rows:
            by_depth.setdefault(int(row["conv_depth"]), {})[str(row["checkpoint_kind"])] = row
        lines.extend(
            [
                "Mean all-unit saturation by checkpoint on the 2048-sample test subset:",
                "",
                "| Conv | Init | Best | Final |",
                "|---|---:|---:|---:|",
            ]
        )
        for depth, group in sorted(by_depth.items()):
            lines.append(
                f"| Conv{depth} | {_pct(group.get('init', {}).get('all_total_saturation_mean'))} "
                f"| {_pct(group.get('best', {}).get('all_total_saturation_mean'))} "
                f"| {_pct(group.get('final', {}).get('all_total_saturation_mean'))} |"
            )
        lines.append("")

    lines.append(END)
    return "\n".join(lines)


def _replace_block(path: Path, block: str) -> None:
    text = path.read_text()
    if START in text and END in text:
        before = text[: text.index(START)]
        after = text[text.index(END) + len(END) :]
        path.write_text(before + block + after)
        return
    insertion = "\n\n## Hopfield EqProp Conv MNIST Saturation\n\n" + block + "\n"
    marker = "\n## Shared Result Roots"
    if marker in text:
        text = text.replace(marker, insertion + marker, 1)
    else:
        text = text.rstrip() + insertion + "\n"
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--current-state", default=str(REPO_ROOT / "docs/current_state.md"))
    parser.add_argument("--final-summary-csv", default=None)
    parser.add_argument("--saturation-csv", default=None)
    parser.add_argument("--saturation-summary-csv", default=None)
    parser.add_argument("--lr-job", default="")
    parser.add_argument("--collect-job", default="")
    parser.add_argument("--final-job", default="")
    parser.add_argument("--saturation-job", default="")
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    args.output_root = str(output_root)
    args.final_summary_csv = args.final_summary_csv or str(output_root / "final_training_summary.csv")
    args.saturation_csv = args.saturation_csv or str(output_root / "final_saturation_train_test_2048.csv")
    args.saturation_summary_csv = args.saturation_summary_csv or str(output_root / "saturation_summary_by_depth_checkpoint_split.csv")
    return args


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    final_rows = _final_rows(output_root / "final")
    final_agg = _aggregate_final(final_rows)
    saturation_agg = _aggregate_saturation(Path(args.saturation_csv))
    _write_csv(Path(args.final_summary_csv), FINAL_COLUMNS, final_rows)
    _write_csv(Path(args.saturation_summary_csv), SAT_COLUMNS, saturation_agg)
    block = _markdown(args, final_rows, final_agg, saturation_agg)
    _replace_block(Path(args.current_state), block)
    print(
        f"[report] final_rows={len(final_rows)} saturation_rows={len(saturation_agg)} "
        f"current_state={args.current_state}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"report_mnist_hopfield_eqprop_current_state.py: {exc}", file=sys.stderr)
        raise
