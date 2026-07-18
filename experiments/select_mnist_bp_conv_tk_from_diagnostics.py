#!/usr/bin/env python3
"""Select per-row Conv MNIST fixed-step T/K from init diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


T_VALUES_DEFAULT = [4, 6, 8, 10, 16, 24, 32, 48]
K_VALUES_DEFAULT = [4, 6, 8, 16, 32, 64]
OUTPUT_COLUMNS = [
    "conv_depth",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "selected_T",
    "selected_K",
    "boundary_selected",
    "selection_status",
    "residual_status",
    "gradient_status",
    "cosine_status",
    "residual_reference_T",
    "residual_selection_rule",
    "residual_p90_threshold",
    "gradient_reference_K",
    "cosine_reference",
    "residual_relative_tolerance",
    "gradient_relative_tolerance",
    "zero_fraction_tolerance",
    "cosine_tolerance",
    "max_residual_p90",
    "max_residual_p90_over_threshold",
    "max_residual_p90_ratio",
    "max_gradient_relative_delta",
    "max_zero_fraction_delta",
    "selected_cosine_mean",
    "best_cosine_mean",
    "residual_modes",
    "source_config_path",
    "residual_root",
    "gradient_root",
    "cosine_root",
]


@dataclass(frozen=True, order=True)
class RowKey:
    conv_depth: int
    non_linearity: str
    run_name: str
    seed: int


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csvs(root: Path, pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.expanduser().resolve().glob(pattern)):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def _source_config_metadata(path: Path) -> tuple[RowKey, dict]:
    config = json.loads(path.read_text())
    model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    model_cfg = {
        **config.get("model_base", {}),
        **config.get("model_overrides", {}).get(model_key, {}),
    }
    key = RowKey(
        conv_depth=len(model_cfg.get("conv_pipeline") or []),
        non_linearity=str(model_cfg.get("non_linearity", "")),
        run_name=path.parent.parent.name,
        seed=int(config.get("seed", 0)),
    )
    return key, {
        "voltage_amp": float(model_cfg.get("voltage_amp", math.nan)),
        "current_amp": float(model_cfg.get("current_amp", math.nan)),
        "source_config_path": str(path),
        "conv_pipeline": model_cfg.get("conv_pipeline") or [],
    }


def _discover_expected(preflight_root: Path) -> dict[RowKey, dict]:
    expected: dict[RowKey, dict] = {}
    for source_config_path in sorted(preflight_root.expanduser().resolve().glob("**/source_config.json")):
        key, meta = _source_config_metadata(source_config_path)
        expected[key] = meta
    if not expected:
        raise SystemExit(f"No source_config.json files found under {preflight_root}.")
    return expected


def _conv_depth_from_model_label(value: str) -> int:
    match = re.search(r"conv(\d+)", str(value))
    return int(match.group(1)) if match else 0


def _key_from_diag_row(
    row: dict,
    *,
    source_config_cache: dict[Path, tuple[RowKey, dict]],
    expected_keys: set[RowKey] | None = None,
) -> RowKey | None:
    checkpoint = row.get("checkpoint_path") or row.get("run_dir") or ""
    if checkpoint:
        path = Path(checkpoint).expanduser()
        source_path = path if path.name == "source_config.json" else path / "source_config.json"
        if source_path.exists():
            source_path = source_path.resolve()
            if source_path not in source_config_cache:
                source_config_cache[source_path] = _source_config_metadata(source_path)
            return source_config_cache[source_path][0]

    conv_depth = _conv_depth_from_model_label(row.get("model_label", ""))
    if conv_depth <= 0:
        conv_depth = _int(row.get("conv_depth"), 0)
    non_linearity = str(row.get("non_linearity", ""))
    run_name = str(row.get("run_name", ""))
    seed = _int(row.get("seed", 0), 0)
    if conv_depth > 0 and not non_linearity and run_name and expected_keys:
        matches = [
            key
            for key in expected_keys
            if key.conv_depth == conv_depth
            and key.run_name == run_name
            and key.seed == seed
        ]
        if len(matches) == 1:
            return matches[0]
    if conv_depth <= 0 or not non_linearity or not run_name:
        return None
    return RowKey(conv_depth, non_linearity, run_name, seed)


def _residual_groups(
    root: Path, expected_keys: set[RowKey]
) -> dict[RowKey, dict[int, dict[tuple[str, str], dict]]]:
    rows = _read_csvs(root, "**/residual_vs_iterations.csv")
    if not rows:
        rows = _read_csvs(root, "**/raw_residual*.csv")
    cache: dict[Path, tuple[RowKey, dict]] = {}
    grouped: dict[RowKey, dict[int, dict[tuple[str, str], dict]]] = {}
    for row in rows:
        key = _key_from_diag_row(
            row, source_config_cache=cache, expected_keys=expected_keys
        )
        if key is None:
            continue
        iteration = _int(row.get("iteration_count"), 0)
        layer_key = (str(row.get("layer_role", "")), str(row.get("layer_name", "")))
        grouped.setdefault(key, {}).setdefault(iteration, {})[layer_key] = row
    return grouped


def _gradient_groups(
    root: Path, expected_keys: set[RowKey]
) -> dict[RowKey, dict[int, dict[int, dict[str, dict]]]]:
    rows = _read_csvs(root, "**/param_summary.csv")
    cache: dict[Path, tuple[RowKey, dict]] = {}
    grouped: dict[RowKey, dict[int, dict[int, dict[str, dict]]]] = {}
    for row in rows:
        param_name = str(row.get("param_name", ""))
        if not param_name.startswith("ConvWeight_"):
            continue
        key = _key_from_diag_row(
            row, source_config_cache=cache, expected_keys=expected_keys
        )
        if key is None:
            continue
        t_value = _int(row.get("t", row.get("T", "")), 0)
        k_value = _int(row.get("k", row.get("K", "")), 0)
        grouped.setdefault(key, {}).setdefault(t_value, {}).setdefault(k_value, {})[param_name] = row
    return grouped


def _cosine_groups(
    root: Path, *, beta: float, expected_keys: set[RowKey]
) -> dict[RowKey, dict[int, dict[int, dict]]]:
    rows = _read_csvs(root, "**/summary_by_amp_k_beta.csv")
    grouped: dict[RowKey, dict[int, dict[int, dict]]] = {}
    cache: dict[Path, tuple[RowKey, dict]] = {}
    for row in rows:
        if row.get("scope") != "aggregate" or row.get("param_name") != "all_params":
            continue
        if abs(_float(row.get("beta")) - beta) > 1e-12:
            continue
        key = _key_from_diag_row(
            row, source_config_cache=cache, expected_keys=expected_keys
        )
        if key is None:
            continue
        free_t = _int(row.get("free_iteration_count", row.get("iteration_count")), 0)
        k_value = _int(row.get("iteration_count"), 0)
        grouped.setdefault(key, {}).setdefault(free_t, {})[k_value] = row
    return grouped


def _ratio(value: float, reference: float, eps: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.inf
    if abs(reference) <= eps:
        return 1.0 if abs(value) <= eps else math.inf
    return value / reference


def _relative_delta(value: float, reference: float, eps: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.inf
    scale = max(abs(value), abs(reference), eps)
    return abs(value - reference) / scale


def _select_t(
    rows_by_t: dict[int, dict[tuple[str, str], dict]],
    *,
    t_values: list[int],
    reference_t: int,
    tolerance: float,
    p90_threshold: float,
    eps: float,
) -> tuple[int, str, float, str]:
    reference_rows = rows_by_t.get(reference_t, {})
    if not reference_rows:
        return reference_t, "missing_residual_reference", math.inf, ""
    modes = sorted({str(row.get("residual_mode", "")) for row in reference_rows.values()})

    if p90_threshold > 0.0:
        reference_values = [_float(row.get("p90")) for row in reference_rows.values()]
        if any(not math.isfinite(value) or value >= p90_threshold for value in reference_values):
            return (
                reference_t,
                "residual_reference_above_threshold",
                max(reference_values) if reference_values else math.inf,
                ";".join(modes),
            )
        best_candidate = reference_t
        best_p90 = math.inf
        for t_value in t_values:
            candidate_rows = rows_by_t.get(t_value, {})
            if not candidate_rows:
                continue
            values = []
            failed = False
            for layer_key in reference_rows:
                row = candidate_rows.get(layer_key)
                if row is None:
                    failed = True
                    break
                value = _float(row.get("p90"))
                values.append(value)
                if not math.isfinite(value) or value >= p90_threshold:
                    failed = True
            max_p90 = max(values) if values else math.inf
            if max_p90 < best_p90:
                best_p90 = max_p90
                best_candidate = t_value
            if not failed:
                return t_value, "residual_below_threshold", max_p90, ";".join(modes)
        return best_candidate, "residual_above_threshold", best_p90, ";".join(modes)

    best_candidate = reference_t
    best_ratio = math.inf
    for t_value in t_values:
        candidate_rows = rows_by_t.get(t_value, {})
        if not candidate_rows:
            continue
        ratios = []
        failed = False
        for layer_key, ref_row in reference_rows.items():
            row = candidate_rows.get(layer_key)
            if row is None:
                failed = True
                break
            ratio = _ratio(_float(row.get("p90")), _float(ref_row.get("p90")), eps)
            ratios.append(ratio)
            if ratio > 1.0 + tolerance:
                failed = True
        max_ratio = max(ratios) if ratios else math.inf
        if max_ratio < best_ratio:
            best_ratio = max_ratio
            best_candidate = t_value
        if not failed:
            return t_value, "residual_settled", max_ratio, ";".join(modes)
    return best_candidate, "residual_not_settled", best_ratio, ";".join(modes)


def _gradient_pass(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    relative_tolerance: float,
    zero_fraction_tolerance: float,
    eps: float,
) -> tuple[bool, float, float]:
    max_delta = 0.0
    max_zero_delta = 0.0
    if not reference:
        return False, math.inf, math.inf
    for param_name, ref_row in reference.items():
        row = candidate.get(param_name)
        if row is None:
            return False, math.inf, math.inf
        delta = _relative_delta(
            _float(row.get("grad_l2_mean")),
            _float(ref_row.get("grad_l2_mean")),
            eps,
        )
        zero_delta = abs(
            _float(row.get("grad_zero_fraction_mean"))
            - _float(ref_row.get("grad_zero_fraction_mean"))
        )
        max_delta = max(max_delta, delta)
        max_zero_delta = max(max_zero_delta, zero_delta)
        if delta > relative_tolerance or zero_delta > zero_fraction_tolerance:
            return False, max_delta, max_zero_delta
    return True, max_delta, max_zero_delta


def _select_k(
    gradients_by_t: dict[int, dict[int, dict[str, dict]]],
    cosines_by_t: dict[int, dict[int, dict]],
    *,
    selected_t: int,
    k_values: list[int],
    reference_k: int,
    gradient_relative_tolerance: float,
    zero_fraction_tolerance: float,
    cosine_tolerance: float,
    eps: float,
) -> tuple[int, str, str, float, float, float, float]:
    gradients_by_k = gradients_by_t.get(selected_t, {})
    reference = gradients_by_k.get(reference_k, {})
    cosines_by_k = cosines_by_t.get(selected_t, {})
    finite_cosines = [
        _float(row.get("cosine_mean"))
        for k, row in cosines_by_k.items()
        if k in k_values and math.isfinite(_float(row.get("cosine_mean")))
    ]
    best_cosine = max(finite_cosines) if finite_cosines else math.nan

    best_k = reference_k
    best_grad_delta = math.inf
    best_zero_delta = math.inf
    best_cosine_at_k = math.nan
    gradient_status = "missing_gradient_reference" if not reference else "gradient_not_settled"
    cosine_status = "missing_cosine_diagnostic"

    def cosine_diagnostic(cosine: float) -> str:
        if not math.isfinite(cosine) or not math.isfinite(best_cosine):
            return "missing_cosine_diagnostic"
        if cosine >= best_cosine - cosine_tolerance:
            return "cosine_within_tolerance_diagnostic"
        return "cosine_below_tolerance_diagnostic"

    for k_value in k_values:
        candidate = gradients_by_k.get(k_value, {})
        grad_ok, grad_delta, zero_delta = _gradient_pass(
            candidate,
            reference,
            relative_tolerance=gradient_relative_tolerance,
            zero_fraction_tolerance=zero_fraction_tolerance,
            eps=eps,
        )
        cosine_row = cosines_by_k.get(k_value)
        cosine = _float(cosine_row.get("cosine_mean")) if cosine_row else math.nan
        if grad_delta < best_grad_delta:
            best_k = k_value
            best_grad_delta = grad_delta
            best_zero_delta = zero_delta
            best_cosine_at_k = cosine
        if grad_ok:
            return (
                k_value,
                "gradient_settled",
                cosine_diagnostic(cosine),
                grad_delta,
                zero_delta,
                cosine,
                best_cosine,
            )

    return (
        best_k,
        gradient_status,
        cosine_diagnostic(best_cosine_at_k),
        best_grad_delta,
        best_zero_delta,
        best_cosine_at_k,
        best_cosine,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", required=True)
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--gradient-root", required=True)
    parser.add_argument("--cosine-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--T-values", type=int, nargs="+", default=T_VALUES_DEFAULT)
    parser.add_argument("--K-values", type=int, nargs="+", default=K_VALUES_DEFAULT)
    parser.add_argument("--reference-T", type=int, default=48)
    parser.add_argument("--reference-K", type=int, default=64)
    parser.add_argument("--residual-p90-threshold", type=float, default=1.0e-2)
    parser.add_argument("--residual-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--gradient-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--zero-fraction-tolerance", type=float, default=0.02)
    parser.add_argument("--cosine-beta", type=float, default=0.25)
    parser.add_argument("--cosine-tolerance", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1.0e-30)
    args = parser.parse_args()

    preflight_root = Path(args.preflight_root).expanduser().resolve()
    residual_root = Path(args.residual_root).expanduser().resolve()
    gradient_root = Path(args.gradient_root).expanduser().resolve()
    cosine_root = Path(args.cosine_root).expanduser().resolve()

    expected = _discover_expected(preflight_root)
    expected_keys = set(expected)
    residual = _residual_groups(residual_root, expected_keys)
    gradients = _gradient_groups(gradient_root, expected_keys)
    cosines = _cosine_groups(
        cosine_root, beta=float(args.cosine_beta), expected_keys=expected_keys
    )

    rows: list[dict] = []
    for key, meta in sorted(expected.items()):
        selected_t, residual_status, residual_metric, residual_modes = _select_t(
            residual.get(key, {}),
            t_values=[int(value) for value in args.T_values],
            reference_t=int(args.reference_T),
            tolerance=float(args.residual_relative_tolerance),
            p90_threshold=float(args.residual_p90_threshold),
            eps=float(args.eps),
        )
        if float(args.residual_p90_threshold) > 0.0:
            residual_rule = "p90_threshold"
            max_residual_p90 = residual_metric
            max_residual_ratio = (
                residual_metric / float(args.residual_p90_threshold)
                if float(args.residual_p90_threshold) > 0.0
                else math.inf
            )
        else:
            residual_rule = "relative_to_reference_T"
            max_residual_p90 = math.nan
            max_residual_ratio = residual_metric
        (
            selected_k,
            gradient_status,
            cosine_status,
            max_gradient_delta,
            max_zero_delta,
            selected_cosine,
            best_cosine,
        ) = _select_k(
            gradients.get(key, {}),
            cosines.get(key, {}),
            selected_t=selected_t,
            k_values=[int(value) for value in args.K_values],
            reference_k=int(args.reference_K),
            gradient_relative_tolerance=float(args.gradient_relative_tolerance),
            zero_fraction_tolerance=float(args.zero_fraction_tolerance),
            cosine_tolerance=float(args.cosine_tolerance),
            eps=float(args.eps),
        )
        boundary = selected_t == max(args.T_values) or selected_k == max(args.K_values)
        status_parts = [residual_status, gradient_status]
        if boundary:
            status_parts.append("boundary_selected")
        rows.append(
            {
                "conv_depth": key.conv_depth,
                "non_linearity": key.non_linearity,
                "run_name": key.run_name,
                "seed": key.seed,
                "voltage_amp": meta.get("voltage_amp", ""),
                "current_amp": meta.get("current_amp", ""),
                "selected_T": selected_t,
                "selected_K": selected_k,
                "boundary_selected": bool(boundary),
                "selection_status": ";".join(status_parts),
                "residual_status": residual_status,
                "gradient_status": gradient_status,
                "cosine_status": cosine_status,
                "residual_reference_T": int(args.reference_T),
                "residual_selection_rule": residual_rule,
                "residual_p90_threshold": float(args.residual_p90_threshold),
                "gradient_reference_K": int(args.reference_K),
                "cosine_reference": "diagnostic_only_best_tested_K_at_selected_T",
                "residual_relative_tolerance": float(args.residual_relative_tolerance),
                "gradient_relative_tolerance": float(args.gradient_relative_tolerance),
                "zero_fraction_tolerance": float(args.zero_fraction_tolerance),
                "cosine_tolerance": float(args.cosine_tolerance),
                "max_residual_p90": max_residual_p90,
                "max_residual_p90_over_threshold": max_residual_ratio,
                "max_residual_p90_ratio": max_residual_ratio,
                "max_gradient_relative_delta": max_gradient_delta,
                "max_zero_fraction_delta": max_zero_delta,
                "selected_cosine_mean": selected_cosine,
                "best_cosine_mean": best_cosine,
                "residual_modes": residual_modes,
                "source_config_path": meta.get("source_config_path", ""),
                "residual_root": str(residual_root),
                "gradient_root": str(gradient_root),
                "cosine_root": str(cosine_root),
            }
        )

    _write_csv(Path(args.output).expanduser().resolve(), rows)
    print(f"[select-tk] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
