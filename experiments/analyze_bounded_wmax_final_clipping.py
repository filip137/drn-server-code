#!/usr/bin/env python3
"""Measure exact final-bound occupancy in collected bounded-wmax runs.

This analyzer is read-only with respect to experiment bundles.  It reuses the
bounded-wmax study join and canonical-bundle validation, then reads the indexed
``weights_final.npz`` checkpoint for each uniquely valid completed arm.  Only
``ConvWeight_*`` and ``DenseWeight_*`` parameters are included; biases and
checkpoint metadata are intentionally excluded.

The primary measurement uses exact equality against ``weight_min`` and
``weight_max`` cast to each checkpoint array's dtype.  A separate ``rtol=0``
tolerance audit is included by default so that representationally adjacent
values cannot be confused with the exact endpoint measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from experiments.analyze_bounded_wmax_sweep import (
        DEFAULT_STUDY_MANIFEST,
        _discover_and_join,
        _load_json,
        _sha256,
        _validate_study_manifest,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from analyze_bounded_wmax_sweep import (  # type: ignore[no-redef]
        DEFAULT_STUDY_MANIFEST,
        _discover_and_join,
        _load_json,
        _sha256,
        _validate_study_manifest,
    )


ANALYSIS_SCHEMA = "perfectdiode-bounded-wmax-final-clipping-analysis/v1"
DEFAULT_AUDIT_ATOL = 1.0e-12
BOUNDED_WEIGHT_PREFIXES = ("ConvWeight_", "DenseWeight_")
SCHEME_COLORS = {
    "baseline": "#4c78a8",
    "ours": "#f58518",
    "legacy": "#54a24b",
}


class IncompleteFinalWeightCoverageError(RuntimeError):
    """Raised after outputs are written when complete coverage was required."""

    def __init__(self, report: Mapping[str, Any]):
        coverage = report["coverage"]
        super().__init__(
            "Final-weight coverage is incomplete or ambiguous: "
            f"usable={coverage['final_weight_usable_arm_count']}/"
            f"{coverage['expected_arm_count']}, "
            f"canonical_usable={coverage['canonical_usable_arm_count']}, "
            f"checkpoint_errors={len(coverage['final_weight_errors'])}."
        )
        self.report = report


def _finite_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return float(value)


def _learning_rate_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configuration = manifest.get("configuration", {})
    resolved = configuration.get("resolved", {}) if isinstance(configuration, Mapping) else {}
    optimizer = resolved.get("optimizer", {}) if isinstance(resolved, Mapping) else {}
    raw = optimizer.get("learning_rate") if isinstance(optimizer, Mapping) else None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        rates = [_finite_float(raw, label="learning rate")]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        rates = [
            _finite_float(value, label=f"learning rate {index}")
            for index, value in enumerate(raw)
        ]
    else:
        raise ValueError(f"Missing scalar or vector learning_rate in {optimizer!r}.")
    if not rates or any(value <= 0.0 for value in rates):
        raise ValueError(f"Learning rates must all be positive, got {rates!r}.")
    return {
        "learning_rate_vector_json": json.dumps(rates, separators=(",", ":")),
        "learning_rate_count": len(rates),
        "learning_rate_min": min(rates),
        "learning_rate_max": max(rates),
    }


def _indexed_final_checkpoint(run_dir: Path) -> Path:
    result = _load_json(run_dir / "result.json")
    artifacts = result.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("result.json artifacts must be a list")
    records = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == "weights_final.npz"
        and item.get("kind") == "checkpoint"
    ]
    if len(records) != 1:
        raise ValueError(
            "Expected exactly one indexed checkpoint named weights_final.npz, "
            f"found {len(records)}."
        )
    relative = Path(str(records[0]["path"]))
    if relative.is_absolute():
        raise ValueError(f"Indexed checkpoint path must be relative: {relative}.")
    checkpoint = (run_dir / relative).resolve()
    try:
        checkpoint.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"Indexed checkpoint escapes its canonical run directory: {relative}."
        ) from error
    if not checkpoint.is_file():
        raise ValueError(f"Indexed final checkpoint is missing: {checkpoint}.")
    return checkpoint


def _parameter_kind(name: str) -> str:
    if name.startswith("ConvWeight_"):
        return "conv_weight"
    if name.startswith("DenseWeight_"):
        return "dense_weight"
    raise ValueError(f"Not a bounded weight parameter: {name!r}.")


def _decode_parameter_name(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("param_names contains invalid UTF-8 bytes") from error
    return str(value)


def _percent(count: int, total: int) -> float:
    return 100.0 * count / total


def _array_endpoint_counts(
    values: np.ndarray,
    *,
    parameter_name: str,
    weight_min: float,
    weight_max: float,
    audit_atol: float | None,
) -> dict[str, Any]:
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(
            f"{parameter_name} must have floating dtype, got {values.dtype}."
        )
    if values.size == 0:
        raise ValueError(f"{parameter_name} is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{parameter_name} contains non-finite values.")

    lower = np.asarray(weight_min, dtype=values.dtype)[()]
    upper = np.asarray(weight_max, dtype=values.dtype)[()]
    if not bool(lower < upper):
        raise ValueError(
            f"{parameter_name} dtype collapses or reverses the configured interval: "
            f"{lower!r} .. {upper!r}."
        )
    exact_lower = values == lower
    exact_upper = values == upper
    exact_either = np.logical_or(exact_lower, exact_upper)
    below = values < lower
    above = values > upper
    outside = np.logical_or(below, above)
    count = int(values.size)
    result: dict[str, Any] = {
        "dtype": str(values.dtype),
        "dtype_weight_min": float(lower),
        "dtype_weight_max": float(upper),
        "num_weights": count,
        "exact_lower_endpoint_count": int(np.count_nonzero(exact_lower)),
        "exact_upper_endpoint_count": int(np.count_nonzero(exact_upper)),
        "exact_either_endpoint_count": int(np.count_nonzero(exact_either)),
        "below_lower_count": int(np.count_nonzero(below)),
        "above_upper_count": int(np.count_nonzero(above)),
        "outside_interval_count": int(np.count_nonzero(outside)),
    }
    for stem in (
        "exact_lower_endpoint",
        "exact_upper_endpoint",
        "exact_either_endpoint",
        "below_lower",
        "above_upper",
        "outside_interval",
    ):
        result[f"{stem}_percent"] = _percent(result[f"{stem}_count"], count)

    result["audit_atol"] = audit_atol
    if audit_atol is None:
        for stem in (
            "audit_lower_endpoint",
            "audit_upper_endpoint",
            "audit_either_endpoint",
            "audit_extra_lower",
            "audit_extra_upper",
            "audit_extra_either",
        ):
            result[f"{stem}_count"] = None
            result[f"{stem}_percent"] = None
        return result

    close_lower = np.isclose(values, lower, rtol=0.0, atol=audit_atol)
    close_upper = np.isclose(values, upper, rtol=0.0, atol=audit_atol)
    close_either = np.logical_or(close_lower, close_upper)
    audit_counts = {
        "audit_lower_endpoint_count": int(np.count_nonzero(close_lower)),
        "audit_upper_endpoint_count": int(np.count_nonzero(close_upper)),
        "audit_either_endpoint_count": int(np.count_nonzero(close_either)),
    }
    result.update(audit_counts)
    result["audit_extra_lower_count"] = (
        audit_counts["audit_lower_endpoint_count"]
        - result["exact_lower_endpoint_count"]
    )
    result["audit_extra_upper_count"] = (
        audit_counts["audit_upper_endpoint_count"]
        - result["exact_upper_endpoint_count"]
    )
    result["audit_extra_either_count"] = (
        audit_counts["audit_either_endpoint_count"]
        - result["exact_either_endpoint_count"]
    )
    for stem in (
        "audit_lower_endpoint",
        "audit_upper_endpoint",
        "audit_either_endpoint",
        "audit_extra_lower",
        "audit_extra_upper",
        "audit_extra_either",
    ):
        result[f"{stem}_percent"] = _percent(result[f"{stem}_count"], count)
    return result


def _aggregate_counts(
    rows: Sequence[Mapping[str, Any]], *, audit_atol: float | None
) -> dict[str, Any]:
    total = sum(int(row["num_weights"]) for row in rows)
    if total <= 0:
        raise ValueError("Cannot pool an empty final-weight checkpoint.")
    dtypes = sorted({str(row["dtype"]) for row in rows})
    dtype_lowers = sorted({float(row["dtype_weight_min"]) for row in rows})
    dtype_uppers = sorted({float(row["dtype_weight_max"]) for row in rows})
    output: dict[str, Any] = {
        "dtype": dtypes[0] if len(dtypes) == 1 else "mixed:" + ",".join(dtypes),
        "dtype_weight_min": dtype_lowers[0] if len(dtype_lowers) == 1 else None,
        "dtype_weight_max": dtype_uppers[0] if len(dtype_uppers) == 1 else None,
        "num_weights": total,
        "audit_atol": audit_atol,
    }
    stems = (
        "exact_lower_endpoint",
        "exact_upper_endpoint",
        "exact_either_endpoint",
        "below_lower",
        "above_upper",
        "outside_interval",
    )
    audit_stems = (
        "audit_lower_endpoint",
        "audit_upper_endpoint",
        "audit_either_endpoint",
        "audit_extra_lower",
        "audit_extra_upper",
        "audit_extra_either",
    )
    for stem in (*stems, *audit_stems):
        values = [row[f"{stem}_count"] for row in rows]
        if any(value is None for value in values):
            output[f"{stem}_count"] = None
            output[f"{stem}_percent"] = None
        else:
            count = sum(int(value) for value in values)
            output[f"{stem}_count"] = count
            output[f"{stem}_percent"] = _percent(count, total)
    return output


def _checkpoint_rows(
    *,
    study_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    audit_atol: float | None,
) -> list[dict[str, Any]]:
    run_dir = Path(str(candidate["run_dir"])).resolve()
    checkpoint = _indexed_final_checkpoint(run_dir)
    metrics = candidate["metrics"]
    learning_rate = _learning_rate_fields(candidate["manifest"])
    common = {
        "global_array_index": int(study_row["global_array_index"]),
        "arm_id": str(study_row["arm_id"]),
        "architecture": str(study_row["architecture"]),
        "scheme": str(study_row["scheme"]),
        "optimizer": str(study_row["optimizer"]).lower(),
        "target": str(study_row["target"]),
        "weight_min": float(study_row["weight_min"]),
        "weight_max": float(study_row["weight_max"]),
        "weight_range_ratio": float(study_row["weight_range_ratio"]),
        **learning_rate,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint),
        "best_epoch": metrics["best_epoch"],
        "best_validation_accuracy": metrics["best_validation_accuracy"],
        "best_validation_accuracy_percent": 100.0
        * metrics["best_validation_accuracy"],
        "final_validation_accuracy": metrics["final_validation_accuracy"],
        "final_validation_accuracy_percent": 100.0
        * metrics["final_validation_accuracy"],
    }

    parameter_rows: list[dict[str, Any]] = []
    with np.load(checkpoint, allow_pickle=False) as payload:
        if "param_names" not in payload.files:
            raise ValueError(f"Checkpoint lacks param_names: {checkpoint}.")
        raw_names = np.asarray(payload["param_names"])
        if raw_names.ndim != 1:
            raise ValueError(f"param_names must be one-dimensional in {checkpoint}.")
        names = [_decode_parameter_name(value) for value in raw_names.tolist()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate param_names in {checkpoint}.")
        bounded_names = [
            name for name in names if name.startswith(BOUNDED_WEIGHT_PREFIXES)
        ]
        if not bounded_names:
            raise ValueError(f"No ConvWeight_* or DenseWeight_* arrays in {checkpoint}.")
        for parameter_index, name in enumerate(bounded_names):
            if name not in payload.files:
                raise ValueError(f"param_names references absent array {name!r}.")
            counts = _array_endpoint_counts(
                np.asarray(payload[name]),
                parameter_name=name,
                weight_min=float(study_row["weight_min"]),
                weight_max=float(study_row["weight_max"]),
                audit_atol=audit_atol,
            )
            parameter_rows.append(
                {
                    **common,
                    "row_level": "parameter",
                    "parameter_index": parameter_index,
                    "parameter_name": name,
                    "parameter_kind": _parameter_kind(name),
                    **counts,
                }
            )

    pooled = {
        **common,
        "row_level": "pooled",
        "parameter_index": None,
        "parameter_name": "__all_bounded_weights__",
        "parameter_kind": "bounded_weight_pool",
        **_aggregate_counts(parameter_rows, audit_atol=audit_atol),
    }
    return [*parameter_rows, pooled]


def _write_csv(
    path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(
    analysis_dir: Path, pooled_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    if not pooled_rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outputs: list[str] = []
    architectures = list(
        dict.fromkeys(str(row["architecture"]) for row in pooled_rows)
    )
    for architecture in architectures:
        architecture_rows = [
            row for row in pooled_rows if row["architecture"] == architecture
        ]
        optimizers = list(
            dict.fromkeys(str(row["optimizer"]) for row in architecture_rows)
        )
        fig, axes = plt.subplots(
            len(optimizers),
            2,
            figsize=(11.5, 3.6 * len(optimizers)),
            squeeze=False,
            sharex="col",
        )
        for optimizer_index, optimizer in enumerate(optimizers):
            selected = [
                row for row in architecture_rows if row["optimizer"] == optimizer
            ]
            accuracy_axis = axes[optimizer_index][0]
            endpoint_axis = axes[optimizer_index][1]
            for scheme in ("baseline", "ours", "legacy"):
                points = sorted(
                    (row for row in selected if row["scheme"] == scheme),
                    key=lambda row: float(row["weight_max"]),
                )
                if not points:
                    continue
                x = [float(row["weight_max"]) for row in points]
                color = SCHEME_COLORS[scheme]
                accuracy_axis.plot(
                    x,
                    [float(row["best_validation_accuracy_percent"]) for row in points],
                    color=color,
                    linestyle="-",
                    marker="o",
                )
                accuracy_axis.plot(
                    x,
                    [float(row["final_validation_accuracy_percent"]) for row in points],
                    color=color,
                    linestyle="--",
                    marker="x",
                )
                endpoint_axis.plot(
                    x,
                    [float(row["exact_lower_endpoint_percent"]) for row in points],
                    color=color,
                    linestyle="-",
                    marker="o",
                )
                endpoint_axis.plot(
                    x,
                    [float(row["exact_upper_endpoint_percent"]) for row in points],
                    color=color,
                    linestyle="--",
                    marker="x",
                )
            for axis in (accuracy_axis, endpoint_axis):
                axis.set_xscale("log")
                axis.set_xlabel("Weight maximum")
                axis.grid(True, alpha=0.25)
            accuracy_axis.set_ylabel("Validation accuracy (%)")
            endpoint_axis.set_ylabel("Final endpoint occupancy (%)")
            accuracy_axis.set_title(f"{optimizer.upper()} · best (solid) / final (dashed)")
            endpoint_axis.set_title(f"{optimizer.upper()} · lower (solid) / upper (dashed)")

        scheme_handles = [
            Line2D([0], [0], color=color, linewidth=2, label=scheme)
            for scheme, color in SCHEME_COLORS.items()
        ]
        fig.legend(
            handles=scheme_handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
        fig.suptitle(
            f"{architecture.upper()} accuracy and final endpoint occupancy",
            y=1.02,
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        filename = f"accuracy_and_final_endpoint_occupancy_{architecture}.png"
        fig.savefig(analysis_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(filename)
    return outputs


def _best_observed_rows(
    pooled_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the observed wmax winner for best and final validation accuracy."""
    by_surface: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pooled_rows:
        key = (str(row["architecture"]), str(row["scheme"]), str(row["optimizer"]))
        by_surface[key].append(row)

    output: list[dict[str, Any]] = []
    for key, candidates in sorted(by_surface.items()):
        for metric in ("best_validation_accuracy", "final_validation_accuracy"):
            # Prefer the smaller interval if two recorded values are exactly tied.
            selected = max(
                candidates,
                key=lambda row: (float(row[metric]), -float(row["weight_max"])),
            )
            output.append(
                {
                    "architecture": key[0],
                    "scheme": key[1],
                    "optimizer": key[2],
                    "selection_metric": metric,
                    "observed_weight_max_count": len(candidates),
                    "selected_weight_min": selected["weight_min"],
                    "selected_weight_max": selected["weight_max"],
                    "selected_accuracy": selected[metric],
                    "selected_accuracy_percent": selected[f"{metric}_percent"],
                    "best_validation_accuracy_percent": selected[
                        "best_validation_accuracy_percent"
                    ],
                    "final_validation_accuracy_percent": selected[
                        "final_validation_accuracy_percent"
                    ],
                    "exact_lower_endpoint_percent": selected[
                        "exact_lower_endpoint_percent"
                    ],
                    "exact_upper_endpoint_percent": selected[
                        "exact_upper_endpoint_percent"
                    ],
                    "exact_either_endpoint_percent": selected[
                        "exact_either_endpoint_percent"
                    ],
                    "num_weights": selected["num_weights"],
                    "learning_rate_vector_json": selected[
                        "learning_rate_vector_json"
                    ],
                    "arm_id": selected["arm_id"],
                    "run_dir": selected["run_dir"],
                }
            )
    return output


def analyze(
    *,
    study_manifest_path: Path,
    result_roots: Sequence[Path],
    analysis_dir: Path,
    audit_atol: float | None = DEFAULT_AUDIT_ATOL,
    require_complete: bool = False,
) -> dict[str, Any]:
    if audit_atol is not None and (
        not math.isfinite(float(audit_atol)) or float(audit_atol) < 0.0
    ):
        raise ValueError(f"audit_atol must be finite and non-negative: {audit_atol!r}.")
    study_manifest_path = Path(study_manifest_path).expanduser().resolve()
    study_manifest = _load_json(study_manifest_path)
    study_rows = _validate_study_manifest(study_manifest, path=study_manifest_path)
    if not result_roots:
        raise ValueError("Expected at least one local result root.")
    resolved_roots = [Path(path).expanduser().resolve() for path in result_roots]
    usable, canonical_coverage = _discover_and_join(
        resolved_roots,
        study_manifest=study_manifest,
        rows=study_rows,
    )

    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for study_row in study_rows:
        arm_id = str(study_row["arm_id"])
        candidate = usable.get(arm_id)
        if candidate is None:
            continue
        try:
            all_rows.extend(
                _checkpoint_rows(
                    study_row=study_row,
                    candidate=candidate,
                    audit_atol=audit_atol,
                )
            )
        except (OSError, ValueError) as error:
            errors.append(
                {
                    "global_array_index": int(study_row["global_array_index"]),
                    "arm_id": arm_id,
                    "run_dir": candidate["run_dir"],
                    "error": str(error),
                }
            )

    parameter_rows = [row for row in all_rows if row["row_level"] == "parameter"]
    pooled_rows = [row for row in all_rows if row["row_level"] == "pooled"]
    final_weight_arm_ids = {str(row["arm_id"]) for row in pooled_rows}
    expected_arm_ids = {str(row["arm_id"]) for row in study_rows}
    coverage = {
        **canonical_coverage,
        "canonical_complete": bool(canonical_coverage["complete"]),
        "canonical_usable_arm_count": int(canonical_coverage["usable_arm_count"]),
        "final_weight_complete": (
            bool(canonical_coverage["complete"])
            and final_weight_arm_ids == expected_arm_ids
            and not errors
        ),
        "final_weight_usable_arm_count": len(final_weight_arm_ids),
        "final_weight_missing_arm_ids": [
            str(row["arm_id"])
            for row in study_rows
            if str(row["arm_id"]) not in final_weight_arm_ids
        ],
        "final_weight_errors": errors,
    }
    coverage["complete"] = coverage["final_weight_complete"]

    analysis_dir = Path(analysis_dir).expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_csv = analysis_dir / "final_weight_endpoint_occupancy.csv"
    pooled_csv = analysis_dir / "accuracy_and_final_weight_endpoint_occupancy.csv"
    fieldnames = list(all_rows[0]) if all_rows else [
        "global_array_index",
        "arm_id",
        "architecture",
        "scheme",
        "optimizer",
        "target",
        "weight_min",
        "weight_max",
        "weight_range_ratio",
        "learning_rate_vector_json",
        "learning_rate_count",
        "learning_rate_min",
        "learning_rate_max",
        "run_dir",
        "checkpoint_path",
        "best_epoch",
        "best_validation_accuracy",
        "best_validation_accuracy_percent",
        "final_validation_accuracy",
        "final_validation_accuracy_percent",
        "row_level",
        "parameter_index",
        "parameter_name",
        "parameter_kind",
        "dtype",
        "dtype_weight_min",
        "dtype_weight_max",
        "num_weights",
        "exact_lower_endpoint_count",
        "exact_lower_endpoint_percent",
        "exact_upper_endpoint_count",
        "exact_upper_endpoint_percent",
        "exact_either_endpoint_count",
        "exact_either_endpoint_percent",
        "below_lower_count",
        "below_lower_percent",
        "above_upper_count",
        "above_upper_percent",
        "outside_interval_count",
        "outside_interval_percent",
        "audit_atol",
        "audit_lower_endpoint_count",
        "audit_lower_endpoint_percent",
        "audit_upper_endpoint_count",
        "audit_upper_endpoint_percent",
        "audit_either_endpoint_count",
        "audit_either_endpoint_percent",
        "audit_extra_lower_count",
        "audit_extra_lower_percent",
        "audit_extra_upper_count",
        "audit_extra_upper_percent",
        "audit_extra_either_count",
        "audit_extra_either_percent",
    ]
    _write_csv(all_csv, fieldnames=fieldnames, rows=all_rows)
    _write_csv(pooled_csv, fieldnames=fieldnames, rows=pooled_rows)
    best_observed_rows = _best_observed_rows(pooled_rows)
    best_csv = analysis_dir / "best_observed_weight_range_by_surface.csv"
    best_fieldnames = list(best_observed_rows[0]) if best_observed_rows else [
        "architecture",
        "scheme",
        "optimizer",
        "selection_metric",
        "observed_weight_max_count",
        "selected_weight_min",
        "selected_weight_max",
        "selected_accuracy",
        "selected_accuracy_percent",
        "best_validation_accuracy_percent",
        "final_validation_accuracy_percent",
        "exact_lower_endpoint_percent",
        "exact_upper_endpoint_percent",
        "exact_either_endpoint_percent",
        "num_weights",
        "learning_rate_vector_json",
        "arm_id",
        "run_dir",
    ]
    _write_csv(best_csv, fieldnames=best_fieldnames, rows=best_observed_rows)
    plot_files = _save_plots(analysis_dir, pooled_rows)

    learning_rates_by_surface: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in pooled_rows:
        key = (str(row["architecture"]), str(row["scheme"]), str(row["optimizer"]))
        learning_rates_by_surface[key].add(str(row["learning_rate_vector_json"]))
    learning_rate_audit = [
        {
            "architecture": key[0],
            "scheme": key[1],
            "optimizer": key[2],
            "observed_learning_rate_vectors": sorted(vectors),
            "fixed_across_collected_wmax": len(vectors) == 1,
        }
        for key, vectors in sorted(learning_rates_by_surface.items())
    ]
    outputs = [
        all_csv.name,
        pooled_csv.name,
        best_csv.name,
        *plot_files,
        "analysis_summary.json",
    ]
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "study_id": study_manifest["study_id"],
        "evidence_class": study_manifest.get("evidence_class"),
        "paper_facing": bool(study_manifest.get("paper_facing", False)),
        "measurement": {
            "included_parameter_prefixes": list(BOUNDED_WEIGHT_PREFIXES),
            "excluded_parameters": "All nonmatching arrays, including Bias_*.",
            "endpoint_definition": (
                "Exact equality to weight_min/weight_max cast to each array dtype."
            ),
            "pooling": "Counts summed over included parameters before percentages.",
            "tolerance_audit": (
                None
                if audit_atol is None
                else {"rtol": 0.0, "atol": float(audit_atol)}
            ),
        },
        "study_manifest": {
            "path": str(study_manifest_path),
            "sha256": _sha256(study_manifest_path),
        },
        "result_roots": [str(path) for path in resolved_roots],
        "coverage": coverage,
        "learning_rate_consistency_by_surface": learning_rate_audit,
        "best_observed_weight_range_by_surface": best_observed_rows,
        "pooled_rows": pooled_rows,
        "parameter_rows": parameter_rows,
        "outputs": outputs,
    }
    summary_path = analysis_dir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if require_complete and not coverage["complete"]:
        raise IncompleteFinalWeightCoverageError(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-manifest", type=Path, default=DEFAULT_STUDY_MANIFEST
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        action="append",
        required=True,
        help="Local collected result root; repeat for multiple roots.",
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-atol",
        type=float,
        default=DEFAULT_AUDIT_ATOL,
        help="Absolute tolerance for the separate rtol=0 audit (default: 1e-12).",
    )
    parser.add_argument(
        "--no-tolerance-audit",
        action="store_true",
        help="Disable the auxiliary tolerance audit; exact counts are unchanged.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write partial outputs, then exit nonzero unless every arm is usable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze(
            study_manifest_path=args.study_manifest,
            result_roots=args.result_root,
            analysis_dir=args.analysis_dir,
            audit_atol=None if args.no_tolerance_audit else args.audit_atol,
            require_complete=args.require_complete,
        )
    except IncompleteFinalWeightCoverageError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "analysis_dir": str(Path(args.analysis_dir).expanduser().resolve()),
                "complete": report["coverage"]["complete"],
                "expected_arm_count": report["coverage"]["expected_arm_count"],
                "study_id": report["study_id"],
                "usable_arm_count": report["coverage"][
                    "final_weight_usable_arm_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
