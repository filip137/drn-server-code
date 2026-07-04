#!/usr/bin/env python3
"""Summarize perfect-diode Conv gradient diagnostics over inference/training K."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def _int(row: dict, key: str) -> int:
    value = _float(row, key)
    return int(value) if math.isfinite(value) else -1


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1e-2:
        return f"{value:.{digits}g}"
    return f"{value:.2e}"


def _alive(row: dict, prefix: str, eps: float) -> str:
    logg = _float(row, f"{prefix}_logg_grad_l2")
    zero_fraction = _float(row, f"{prefix}_grad_zero_fraction")
    if math.isfinite(logg) and logg > eps and (not math.isfinite(zero_fraction) or zero_fraction < 0.999999):
        return "alive"
    if (not math.isfinite(logg) or logg <= eps) and math.isfinite(zero_fraction) and zero_fraction >= 0.999999:
        return "dead"
    if math.isfinite(logg) and logg > eps:
        return "mixed"
    return "zero"


def _run_label(row: dict) -> str:
    depth = _int(row, "conv_depth")
    run_name = row.get("run_name", "")
    gain = _fmt(_float(row, "input_gain"), digits=5)
    lr = _fmt(_float(row, "lr_reference"), digits=5)
    epochs = row.get("epochs", "")
    return f"conv{depth} {run_name} gain={gain} lr={lr} ep={epochs}"


def _read_rows(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    pattern = re.compile(r"T(?P<T>\d+)_K(?P<K>\d+)")
    for csv_path in sorted(input_root.glob("*gradient_T*_K*_init_final_*gpu.csv")):
        match = pattern.search(csv_path.name)
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if match:
                    row.setdefault("inference_iterations_used", match.group("T"))
                    row.setdefault("training_iterations_used", match.group("K"))
                row["source_csv"] = str(csv_path)
                rows.append(row)
    return rows


def _summary_rows(rows: list[dict], phase: str, eps: float) -> list[dict]:
    filtered = [row for row in rows if phase == "all" or row.get("phase") == phase]
    summary = []
    for row in filtered:
        w0 = _alive(row, "weight0", eps)
        w1 = _alive(row, "weight1", eps)
        w2 = _alive(row, "weight2", eps)
        early = "alive" if w0 == "alive" and w1 == "alive" else "dead" if w0 in {"dead", "zero"} and w1 in {"dead", "zero"} else "partial"
        summary.append(
            {
                "conv_depth": str(_int(row, "conv_depth")),
                "run_name": row.get("run_name", ""),
                "phase": row.get("phase", ""),
                "T": str(_int(row, "inference_iterations_used")),
                "K": str(_int(row, "training_iterations_used")),
                "input_gain": _fmt(_float(row, "input_gain"), digits=5),
                "lr_reference": _fmt(_float(row, "lr_reference"), digits=5),
                "best_test_accuracy": _fmt(_float(row, "best_test_accuracy"), digits=5),
                "final_test_accuracy": _fmt(_float(row, "final_test_accuracy"), digits=5),
                "hidden1_saturation": _fmt(_float(row, "hidden1_saturation"), digits=4),
                "hidden2_saturation": _fmt(_float(row, "hidden2_saturation"), digits=4),
                "all_hidden_saturation": _fmt(_float(row, "all_hidden_saturation"), digits=4),
                "global_relative_grad_l2": _fmt(_float(row, "global_relative_grad_l2"), digits=4),
                "weight0_logg_grad_l2": _fmt(_float(row, "weight0_logg_grad_l2"), digits=4),
                "weight1_logg_grad_l2": _fmt(_float(row, "weight1_logg_grad_l2"), digits=4),
                "weight2_logg_grad_l2": _fmt(_float(row, "weight2_logg_grad_l2"), digits=4),
                "weight0_status": w0,
                "weight1_status": w1,
                "weight2_status": w2,
                "early_conv_status": early,
                "run_label": _run_label(row),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            int(row["conv_depth"]),
            row["run_name"],
            float(row["input_gain"] or "nan"),
            row["phase"],
            int(row["T"]),
            int(row["K"]),
        ),
    )


def _write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        output_csv.write_text("")
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict], output_md: Path, phase: str) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Perfect-Diode T/K Gradient Tail Sweep",
        "",
        f"Phase summarized: `{phase}`.",
        "",
        "Status is based on log-conductance gradient `G * dL/dG`; `alive` means nonzero early-conv signal on this diagnostic.",
        "",
    ]
    by_run: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_run[row["run_label"]].append(row)
    for run_label in sorted(by_run):
        lines.extend(
            [
                f"## {run_label}",
                "",
                "| T | K | early | W0 | W1 | W2 | global rel grad | h1 | h2 | all |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in by_run[run_label]:
            lines.append(
                "| {T} | {K} | {early_conv_status} | {weight0_logg_grad_l2} | "
                "{weight1_logg_grad_l2} | {weight2_logg_grad_l2} | "
                "{global_relative_grad_l2} | {hidden1_saturation} | "
                "{hidden2_saturation} | {all_hidden_saturation} |".format(**row)
            )
        lines.append("")
    output_md.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--markdown-output")
    parser.add_argument("--phase", choices=("init", "final", "all"), default="final")
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(Path(args.input_root))
    summary = _summary_rows(rows, phase=args.phase, eps=args.eps)
    _write_csv(summary, Path(args.output_csv))
    if args.markdown_output:
        _write_markdown(summary, Path(args.markdown_output), args.phase)
    print(f"read_rows={len(rows)} summary_rows={len(summary)}")


if __name__ == "__main__":
    main()
