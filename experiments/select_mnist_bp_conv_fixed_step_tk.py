#!/usr/bin/env python3
"""Select shared fixed-step T/K settings from Conv MNIST diagnostic summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


DEFAULT_RUN_NAMES = (
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v4_c0p25",
)

COLUMNS = [
    "conv_depth",
    "strides",
    "paddings",
    "non_linearity",
    "selected_T",
    "selected_K",
    "selection_status",
    "selection_mode",
    "reference_T",
    "reference_K",
    "min_T_floor",
    "t_reference_T",
    "t_reference_K",
    "k_reference_T",
    "k_reference_K",
    "settle_relative_tolerance",
    "settle_zero_fraction_tolerance",
    "max_relative_delta",
    "max_zero_fraction_delta",
    "max_T_relative_delta",
    "max_T_zero_fraction_delta",
    "max_K_relative_delta",
    "max_K_zero_fraction_delta",
    "settled_metric_count",
    "num_expected_runs",
    "num_present_runs",
    "num_alive_runs",
    "median_first_weight_grad_l2",
    "median_global_grad_l2",
    "needs_T64_sentinel",
    "needs_K32_sentinel",
    "source_rows",
]


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("**/run_summary.csv")):
        path_T, path_K = _tk_from_path(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                if "T" not in row or row.get("T", "") == "":
                    row["T"] = row.get("num_iterations_inference", "") or path_T
                if "K" not in row or row.get("K", "") == "":
                    row["K"] = row.get("num_iterations_training", "") or path_K
                row["source_rows"] = str(path)
                rows.append(row)
    return rows


def _tk_from_path(path: Path) -> tuple[str, str]:
    for part in reversed(path.parts):
        pieces = part.split("_")
        if len(pieces) == 4 and pieces[0] == "T" and pieces[2] == "K":
            return pieces[1], pieces[3]
    return "", ""


def _read_param_rows(root: Path, metadata_rows: list[dict]) -> list[dict]:
    metadata: dict[tuple[int, int, str], dict] = {}
    for row in metadata_rows:
        T = int(float(row.get("T", row.get("num_iterations_inference", 0))))
        K = int(float(row.get("K", row.get("num_iterations_training", 0))))
        metadata[(T, K, str(Path(row["run_dir"]).resolve()))] = row

    rows: list[dict] = []
    for path in sorted(root.glob("**/param_summary.csv")):
        parts = path.parent.name.split("_")
        if len(parts) != 4 or parts[0] != "T" or parts[2] != "K":
            continue
        T = int(parts[1])
        K = int(parts[3])
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not str(row.get("param_name", "")).startswith("ConvWeight_"):
                    continue
                row = dict(row)
                key = (T, K, str(Path(row["run_dir"]).resolve()))
                meta = metadata.get(key)
                if meta is None:
                    continue
                row.update(
                    {
                        "T": T,
                        "K": K,
                        "conv_depth": meta.get("conv_depth", ""),
                        "strides": meta.get("strides", ""),
                        "paddings": meta.get("paddings", ""),
                        "non_linearity": meta.get("non_linearity", ""),
                        "run_name": meta.get("run_name", ""),
                        "source_rows": str(path),
                    }
                )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _alive(row: dict, grad_eps: float) -> bool:
    first = _float(row.get("first_weight_grad_l2_mean"))
    global_grad = _float(row.get("global_grad_l2_mean"))
    return math.isfinite(first) and first > grad_eps and math.isfinite(global_grad) and global_grad > grad_eps


def _relative_delta(value: float, reference: float, eps: float) -> float:
    scale = max(abs(value), abs(reference), eps)
    return abs(value - reference) / scale


def _candidate_summary(
    rows: list[dict],
    *,
    expected_run_names: set[str],
    grad_eps: float,
) -> dict:
    focused = [row for row in rows if row.get("run_name") in expected_run_names]
    alive_rows = [row for row in focused if _alive(row, grad_eps)]
    first_grads = [_float(row.get("first_weight_grad_l2_mean")) for row in focused]
    global_grads = [_float(row.get("global_grad_l2_mean")) for row in focused]
    first_grads = [value for value in first_grads if math.isfinite(value)]
    global_grads = [value for value in global_grads if math.isfinite(value)]
    return {
        "num_present_runs": len({row.get("run_name") for row in focused}),
        "num_alive_runs": len({row.get("run_name") for row in alive_rows}),
        "median_first_weight_grad_l2": median(first_grads) if first_grads else math.nan,
        "median_global_grad_l2": median(global_grads) if global_grads else math.nan,
        "source_rows": ";".join(sorted({row.get("source_rows", "") for row in focused})),
    }


def _settled_candidate_summary(
    rows: list[dict],
    reference_rows: list[dict],
    *,
    expected_run_names: set[str],
    grad_eps: float,
    relative_tolerance: float,
    zero_fraction_tolerance: float,
) -> dict:
    focused = [
        row
        for row in rows
        if row.get("run_name") in expected_run_names
        and str(row.get("param_name", "")).startswith("ConvWeight_")
    ]
    reference = [
        row
        for row in reference_rows
        if row.get("run_name") in expected_run_names
        and str(row.get("param_name", "")).startswith("ConvWeight_")
    ]
    by_metric = {
        (row["run_name"], row["param_name"]): _float(row.get("grad_l2_mean"))
        for row in focused
    }
    by_zero_fraction = {
        (row["run_name"], row["param_name"]): _float(row.get("grad_zero_fraction_mean"))
        for row in focused
    }
    reference_by_metric = {
        (row["run_name"], row["param_name"]): _float(row.get("grad_l2_mean"))
        for row in reference
    }
    reference_by_zero_fraction = {
        (row["run_name"], row["param_name"]): _float(row.get("grad_zero_fraction_mean"))
        for row in reference
    }
    run_names_present = {row.get("run_name") for row in focused}
    alive_runs = {
        row.get("run_name")
        for row in focused
        if _float(row.get("grad_l2_mean")) > grad_eps
    }
    deltas = []
    zero_fraction_deltas = []
    settled = True
    for key, ref_value in reference_by_metric.items():
        value = by_metric.get(key, math.nan)
        if not math.isfinite(value) or not math.isfinite(ref_value) or value <= grad_eps:
            settled = False
            continue
        delta = _relative_delta(value, ref_value, grad_eps)
        deltas.append(delta)
        if delta > relative_tolerance:
            settled = False
        zero_fraction = by_zero_fraction.get(key, math.nan)
        reference_zero_fraction = reference_by_zero_fraction.get(key, math.nan)
        if math.isfinite(zero_fraction) and math.isfinite(reference_zero_fraction):
            zero_fraction_delta = abs(zero_fraction - reference_zero_fraction)
            zero_fraction_deltas.append(zero_fraction_delta)
            if zero_fraction_delta > zero_fraction_tolerance:
                settled = False
    return {
        "settled": bool(
            settled
            and run_names_present == expected_run_names
            and expected_run_names.issubset(alive_runs)
            and len(deltas) > 0
        ),
        "num_present_runs": len(run_names_present),
        "num_alive_runs": len(expected_run_names.intersection(alive_runs)),
        "max_relative_delta": max(deltas) if deltas else math.nan,
        "max_zero_fraction_delta": max(zero_fraction_deltas) if zero_fraction_deltas else math.nan,
        "settled_metric_count": len(deltas),
        "source_rows": ";".join(sorted({row.get("source_rows", "") for row in focused})),
    }


def _select_alive(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    expected = set(args.expected_run_names)
    grouped: dict[tuple[int, str, str, str], dict[tuple[int, int], list[dict]]] = {}
    for row in rows:
        non_linearity = row.get("non_linearity")
        if not non_linearity:
            continue
        key = (
            int(float(row.get("conv_depth", 0))),
            str(row.get("strides", "")),
            str(row.get("paddings", "")),
            str(non_linearity),
        )
        T = int(float(row.get("num_iterations_inference", 0)))
        K = int(float(row.get("num_iterations_training", 0)))
        grouped.setdefault(key, {}).setdefault((T, K), []).append(row)

    out: list[dict] = []
    for (conv_depth, strides, paddings, non_linearity), by_tk in sorted(grouped.items()):
        candidates = []
        for (T, K), candidate_rows in sorted(by_tk.items()):
            summary = _candidate_summary(
                candidate_rows,
                expected_run_names=expected,
                grad_eps=float(args.grad_eps),
            )
            full_and_alive = (
                summary["num_present_runs"] == len(expected)
                and summary["num_alive_runs"] == len(expected)
            )
            candidates.append((full_and_alive, T, K, summary))

        valid = [item for item in candidates if item[0]]
        if valid:
            _ok, T, K, summary = sorted(valid, key=lambda item: (item[1], item[2]))[0]
            status = "all_expected_gradients_alive"
        else:
            _ok, T, K, summary = sorted(
                candidates,
                key=lambda item: (
                    -int(item[3]["num_alive_runs"]),
                    -int(item[3]["num_present_runs"]),
                    -_float(item[3]["median_first_weight_grad_l2"], -math.inf),
                    item[1],
                    item[2],
                ),
            )[0]
            status = "fallback_max_alive"

        out.append(
            {
                "conv_depth": conv_depth,
                "strides": strides,
                "paddings": paddings,
                "non_linearity": non_linearity,
                "selected_T": T,
                "selected_K": K,
                "selection_status": status,
                "selection_mode": "alive",
                "reference_T": "",
                "reference_K": "",
                "settle_relative_tolerance": "",
                "settle_zero_fraction_tolerance": "",
                "max_relative_delta": "",
                "max_zero_fraction_delta": "",
                "settled_metric_count": "",
                "num_expected_runs": len(expected),
                **summary,
                "needs_T64_sentinel": bool(T == max(args.T_values)),
                "needs_K32_sentinel": bool(K == max(args.K_values)),
            }
        )
    return out


def _select_settled(rows: list[dict], param_rows: list[dict], args: argparse.Namespace) -> list[dict]:
    expected = set(args.expected_run_names)
    min_t_by_depth = _parse_min_t_by_conv_depth(args.min_T_by_conv_depth)
    grouped: dict[tuple[int, str, str, str], dict[tuple[int, int], list[dict]]] = {}
    for row in param_rows:
        non_linearity = row.get("non_linearity")
        if not non_linearity:
            continue
        key = (
            int(float(row.get("conv_depth", 0))),
            str(row.get("strides", "")),
            str(row.get("paddings", "")),
            str(non_linearity),
        )
        T = int(float(row.get("T", 0)))
        K = int(float(row.get("K", 0)))
        grouped.setdefault(key, {}).setdefault((T, K), []).append(row)

    run_rows_by_group_tk: dict[tuple[int, str, str, str, int, int], list[dict]] = {}
    for row in rows:
        key = (
            int(float(row.get("conv_depth", 0))),
            str(row.get("strides", "")),
            str(row.get("paddings", "")),
            str(row.get("non_linearity", "")),
            int(float(row.get("T", 0))),
            int(float(row.get("K", 0))),
        )
        run_rows_by_group_tk.setdefault(key, []).append(row)

    out: list[dict] = []
    for (conv_depth, strides, paddings, non_linearity), by_tk in sorted(grouped.items()):
        standard_k_values = {int(value) for value in args.K_values}
        standard_ks_present = sorted({K for _T, K in by_tk if K in standard_k_values})
        t_reference_K = standard_ks_present[-1] if standard_ks_present else max(K for _T, K in by_tk)
        min_T_floor = min_t_by_depth.get(conv_depth, min(args.T_values))
        t_candidates = sorted(T for T, K in by_tk if K == t_reference_K and T >= min_T_floor)
        if not t_candidates:
            t_candidates = sorted(T for T, K in by_tk if K == t_reference_K)
        t_reference_T = max(t_candidates)
        t_reference_rows = by_tk[(t_reference_T, t_reference_K)]

        t_items = []
        for T in t_candidates:
            candidate_rows = by_tk[(T, t_reference_K)]
            summary = _settled_candidate_summary(
                candidate_rows,
                t_reference_rows,
                expected_run_names=expected,
                grad_eps=float(args.grad_eps),
                relative_tolerance=float(args.settle_relative_tolerance),
                zero_fraction_tolerance=float(args.settle_zero_fraction_tolerance),
            )
            t_items.append((summary["settled"], T, summary))

        valid_t = [item for item in t_items if item[0]]
        if valid_t:
            _ok, selected_T, t_summary = sorted(valid_t, key=lambda item: item[1])[0]
            t_status = "T_settled"
        else:
            _ok, selected_T, t_summary = sorted(
                t_items,
                key=lambda item: (
                    -int(item[2]["num_alive_runs"]),
                    _float(item[2]["max_relative_delta"], math.inf),
                    item[1],
                ),
            )[0]
            t_status = "fallback_T_not_settled"

        k_candidates = sorted(K for T, K in by_tk if T == selected_T)
        k_reference_T = selected_T
        k_reference_K = max(k_candidates)
        k_reference_rows = by_tk[(k_reference_T, k_reference_K)]
        k_items = []
        for K in k_candidates:
            candidate_rows = by_tk[(selected_T, K)]
            summary = _settled_candidate_summary(
                candidate_rows,
                k_reference_rows,
                expected_run_names=expected,
                grad_eps=float(args.grad_eps),
                relative_tolerance=float(args.settle_relative_tolerance),
                zero_fraction_tolerance=float(args.settle_zero_fraction_tolerance),
            )
            k_items.append((summary["settled"], K, summary))

        valid_k = [item for item in k_items if item[0]]
        if valid_k:
            _ok, selected_K, k_summary = sorted(valid_k, key=lambda item: item[1])[0]
            k_status = "K_settled"
        else:
            _ok, selected_K, k_summary = sorted(
                k_items,
                key=lambda item: (
                    -int(item[2]["num_alive_runs"]),
                    _float(item[2]["max_relative_delta"], math.inf),
                    item[1],
                ),
            )[0]
            k_status = "fallback_K_not_settled"

        run_summary = _candidate_summary(
            run_rows_by_group_tk.get(
                (conv_depth, strides, paddings, non_linearity, selected_T, selected_K), []
            ),
            expected_run_names=expected,
            grad_eps=float(args.grad_eps),
        )
        max_relative_delta = max(
            _float(t_summary.get("max_relative_delta"), -math.inf),
            _float(k_summary.get("max_relative_delta"), -math.inf),
        )
        max_zero_fraction_delta = max(
            _float(t_summary.get("max_zero_fraction_delta"), -math.inf),
            _float(k_summary.get("max_zero_fraction_delta"), -math.inf),
        )
        status = (
            "convweight_gradients_settled_vs_reference"
            if t_status == "T_settled" and k_status == "K_settled"
            else f"{t_status};{k_status}"
        )
        k32_sentinel_value = max(args.K_values) * 2

        out.append(
            {
                "conv_depth": conv_depth,
                "strides": strides,
                "paddings": paddings,
                "non_linearity": non_linearity,
                "selected_T": selected_T,
                "selected_K": selected_K,
                "selection_status": status,
                "selection_mode": "settled_two_stage",
                "reference_T": t_reference_T,
                "reference_K": k_reference_K,
                "min_T_floor": min_T_floor,
                "t_reference_T": t_reference_T,
                "t_reference_K": t_reference_K,
                "k_reference_T": k_reference_T,
                "k_reference_K": k_reference_K,
                "settle_relative_tolerance": float(args.settle_relative_tolerance),
                "settle_zero_fraction_tolerance": float(args.settle_zero_fraction_tolerance),
                "max_relative_delta": max_relative_delta,
                "max_zero_fraction_delta": max_zero_fraction_delta,
                "max_T_relative_delta": t_summary.get("max_relative_delta", ""),
                "max_T_zero_fraction_delta": t_summary.get("max_zero_fraction_delta", ""),
                "max_K_relative_delta": k_summary.get("max_relative_delta", ""),
                "max_K_zero_fraction_delta": k_summary.get("max_zero_fraction_delta", ""),
                "settled_metric_count": int(t_summary.get("settled_metric_count", 0))
                + int(k_summary.get("settled_metric_count", 0)),
                "num_expected_runs": len(expected),
                "num_present_runs": run_summary.get("num_present_runs", ""),
                "num_alive_runs": run_summary.get("num_alive_runs", ""),
                "median_first_weight_grad_l2": run_summary.get("median_first_weight_grad_l2", ""),
                "median_global_grad_l2": run_summary.get("median_global_grad_l2", ""),
                "needs_T64_sentinel": bool(
                    selected_T == max(args.T_values) and (max(args.T_values) * 2, selected_K) not in by_tk
                ),
                "needs_K32_sentinel": bool(
                    selected_K == max(args.K_values) and (selected_T, k32_sentinel_value) not in by_tk
                ),
                "source_rows": ";".join(
                    sorted(
                        {
                            str(t_summary.get("source_rows", "")),
                            str(k_summary.get("source_rows", "")),
                        }
                    )
                ),
            }
        )
    return out


def select(rows: list[dict], param_rows: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.selection_mode == "alive":
        return _select_alive(rows, args)
    return _select_settled(rows, param_rows, args)


def _parse_min_t_by_conv_depth(values: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for value in values:
        if ":" not in value:
            raise SystemExit(
                "Expected --min-T-by-conv-depth values as DEPTH:T, "
                f"got {value!r}."
            )
        depth, min_t = value.split(":", 1)
        out[int(depth)] = int(min_t)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--T-values", type=int, nargs="+", default=[4, 6, 8, 10, 16, 32])
    parser.add_argument("--K-values", type=int, nargs="+", default=[4, 6, 8, 16])
    parser.add_argument("--expected-run-names", nargs="+", default=list(DEFAULT_RUN_NAMES))
    parser.add_argument("--grad-eps", type=float, default=1e-12)
    parser.add_argument("--selection-mode", choices=("settled", "alive"), default="settled")
    parser.add_argument("--settle-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--settle-zero-fraction-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--min-T-by-conv-depth",
        nargs="*",
        default=[],
        help="Optional conservative T floors as DEPTH:T, e.g. 2:6 3:10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(Path(args.diagnostics_root).expanduser().resolve())
    if not rows:
        raise SystemExit(f"No run_summary.csv files found under {args.diagnostics_root}.")
    param_rows = _read_param_rows(Path(args.diagnostics_root).expanduser().resolve(), rows)
    if args.selection_mode == "settled" and not param_rows:
        raise SystemExit(
            f"No ConvWeight param_summary.csv rows found under {args.diagnostics_root}."
        )
    selected = select(rows, param_rows, args)
    _write_csv(Path(args.output).expanduser().resolve(), selected)
    print(f"[select-tk] groups={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
