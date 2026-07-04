#!/usr/bin/env python3
"""Run a clean hard-sigmoid dense BP gradient sweep over separate T and K."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_mnist_bp_conv_good_run_gradient_saturation import (  # noqa: E402
    _load_specs,
    _measure_phase,
    _normalize_run_dir,
)


DENSE_PREFIXES = ("weight0", "weight1", "weight2")

OUTPUT_COLUMNS = [
    "conv_depth",
    "run_name",
    "phase",
    "T",
    "K",
    "adaptive_equilibrium",
    "voltage_amp",
    "current_amp",
    "input_gain",
    "v_off",
    "best_test_accuracy",
    "final_test_accuracy",
    "hidden1_saturation",
    "hidden2_saturation",
    "all_hidden_saturation",
    "global_relative_grad_l2",
    "weight0_grad_l2",
    "weight0_grad_zero_fraction",
    "weight0_status",
    "weight1_grad_l2",
    "weight1_grad_zero_fraction",
    "weight1_status",
    "weight2_grad_l2",
    "weight2_grad_zero_fraction",
    "weight2_status",
    "input_dense_status",
    "all_dense_status",
    "run_dir",
]


def _fmt(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    if abs(numeric) >= 1e-2:
        return f"{numeric:.{digits}g}"
    return f"{numeric:.2e}"


def _status(grad_l2: float, zero_fraction: float, eps: float) -> str:
    if math.isfinite(grad_l2) and grad_l2 > eps and (
        not math.isfinite(zero_fraction) or zero_fraction < 0.999999
    ):
        return "alive"
    if math.isfinite(zero_fraction) and zero_fraction >= 0.999999:
        return "dead"
    if math.isfinite(grad_l2) and grad_l2 > eps:
        return "mixed"
    return "zero"


def _param_by_name(row: dict) -> dict[str, dict]:
    summaries = json.loads(row.get("param_summaries_json") or "[]")
    by_name = {}
    for summary in summaries:
        name = str(summary.get("name", "")).strip()
        if name:
            by_name[name] = summary
    return by_name


def _add_dense_weight_columns(row: dict, eps: float) -> dict:
    by_name = _param_by_name(row)
    out = dict(row)
    statuses = []
    for index, prefix in enumerate(DENSE_PREFIXES):
        summary = by_name.get(f"DenseWeight_{index}")
        grad_l2 = float(summary.get("grad_l2", math.nan)) if summary else math.nan
        zero_fraction = (
            float(summary.get("grad_zero_fraction", math.nan)) if summary else math.nan
        )
        status = _status(grad_l2, zero_fraction, eps)
        out[f"{prefix}_grad_l2"] = grad_l2
        out[f"{prefix}_grad_zero_fraction"] = zero_fraction
        out[f"{prefix}_status"] = status
        if summary is not None:
            statuses.append(status)
    out["input_dense_status"] = out.get("weight0_status", "missing")
    if statuses and all(status == "alive" for status in statuses):
        out["all_dense_status"] = "alive"
    elif statuses and all(status in {"dead", "zero"} for status in statuses):
        out["all_dense_status"] = "dead"
    elif statuses:
        out["all_dense_status"] = "partial"
    else:
        out["all_dense_status"] = "missing"
    return out


def _compact_row(row: dict) -> dict:
    return {column: row.get(column, "") for column in OUTPUT_COLUMNS}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_compact_row(row))


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Hard-Sigmoid Dense T/K Gradient Sweep",
        "",
        "`W0` is the input-to-hidden DenseWeight. For one-hidden-layer runs, `W1` is the hidden-to-output readout DenseWeight.",
        "",
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["run_name"]), str(row["phase"])), []).append(row)
    for (run_name, phase), group in sorted(grouped.items()):
        lines.extend(
            [
                f"## {run_name} / {phase}",
                "",
                "| T | K | input W | all dense | W0 L2 | W0 zero | W1 L2 | W1 zero | h1 sat | global rel |",
                "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(group, key=lambda item: (int(item["T"]), int(item["K"]))):
            lines.append(
                "| {T} | {K} | {input_status} | {all_status} | {w0} | {z0} | {w1} | {z1} | {h1} | {glob} |".format(
                    T=row["T"],
                    K=row["K"],
                    input_status=row["input_dense_status"],
                    all_status=row["all_dense_status"],
                    w0=_fmt(row["weight0_grad_l2"]),
                    z0=_fmt(row["weight0_grad_zero_fraction"]),
                    w1=_fmt(row["weight1_grad_l2"]),
                    z1=_fmt(row["weight1_grad_zero_fraction"]),
                    h1=_fmt(row["hidden1_saturation"]),
                    glob=_fmt(row["global_relative_grad_l2"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--markdown-output")
    parser.add_argument("--T-values", type=int, nargs="+", required=True)
    parser.add_argument("--K-values", type=int, nargs="+", required=True)
    parser.add_argument("--phase", choices=("init", "final", "both"), default="both")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--perfect-eps", type=float, default=1e-8)
    parser.add_argument(
        "--adaptive-equilibrium",
        action="store_true",
        help="Allow adaptive early stopping; default is exact fixed T/K iterations.",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [_normalize_run_dir(path) for path in args.run_dir]
    specs = _load_specs(run_dirs)
    phases = ["init", "final"] if args.phase == "both" else [args.phase]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = []
    total = len(specs) * len(phases) * len(args.T_values) * len(args.K_values)
    count = 0
    for spec in specs:
        if spec.model_cfg.get("non_linearity") != "hard_sigmoid":
            raise ValueError(
                f"Expected hard_sigmoid run, got {spec.model_cfg.get('non_linearity')}: {spec.run_dir}"
            )
        if spec.model_cfg.get("conv_pipeline") not in (None, []):
            raise ValueError(f"Expected fully connected run with no conv pipeline: {spec.run_dir}")
        for phase in phases:
            for T in args.T_values:
                for K in args.K_values:
                    count += 1
                    print(
                        f"[measure] {count}/{total} {spec.run_dir.parent.name} "
                        f"phase={phase} T={T} K={K}",
                        flush=True,
                    )
                    measure_args = SimpleNamespace(
                        batch_size=args.batch_size,
                        no_download=args.no_download,
                        dataset_root=args.dataset_root,
                        inference_iterations=T,
                        training_iterations=K,
                        split=args.split,
                        max_samples=args.max_samples,
                        perfect_eps=args.perfect_eps,
                        adaptive_equilibrium=args.adaptive_equilibrium,
                    )
                    row = _measure_phase(spec, phase=phase, args=measure_args, device=device)
                    row["T"] = int(T)
                    row["K"] = int(K)
                    row["adaptive_equilibrium"] = bool(args.adaptive_equilibrium)
                    rows.append(_add_dense_weight_columns(row, args.eps))
    _write_csv(Path(args.output_csv), rows)
    if args.markdown_output:
        _write_markdown(Path(args.markdown_output), rows)
    print(f"[done] wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
