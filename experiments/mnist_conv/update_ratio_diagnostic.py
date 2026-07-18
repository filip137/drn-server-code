"""Read-only checkpoint-update diagnostics for canonical MNIST Conv bundles.

This module deliberately consumes completed bundles.  It never rebuilds a
model from scientific command-line flags and never launches training.

``symmetric_relative_delta`` is the bounded checkpoint distance
``||final - best|| / max(||best||, ||final||, 1e-30)``.  It is not a gradient
update estimate and is named explicitly to avoid that scientific ambiguity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import atomic_write_csv, atomic_write_json, read_json, relative_posix
from .layout import ResultLayout
from .manifest import load_manifest
from .runner import InvalidBundleError, validate_bundle
from .specs import RunSpec


DIAGNOSTIC_SCHEMA_VERSION = "mnist-conv-checkpoint-update-ratio/v1"
EPS = 1e-30
PARAMETER_COLUMNS = [
    "run_id", "label", "protocol_id", "category", "architecture",
    "non_linearity", "seed", "voltage_amp", "current_amp", "input_gain",
    "inference_iterations", "training_iterations", "batch_size", "epochs",
    "parameter", "element_count", "best_l2", "final_l2", "delta_l2",
    "symmetric_relative_delta", "cosine", "best_zero_fraction",
    "final_zero_fraction", "run_bundle",
]
SUMMARY_COLUMNS = [
    "run_id", "label", "protocol_id", "category", "architecture",
    "non_linearity", "seed", "voltage_amp", "current_amp", "input_gain",
    "inference_iterations", "training_iterations", "batch_size", "epochs",
    "best_epoch", "best_test_accuracy", "final_test_accuracy",
    "parameter_tensor_count", "parameter_scalar_count",
    "max_symmetric_relative_delta", "mean_symmetric_relative_delta",
    "min_cosine", "final_learning_rate", "run_bundle",
]


class DiagnosticInputError(ValueError):
    """Raised when canonical diagnostic inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class BundleRecord:
    run_id: str
    run_dir: Path
    spec: RunSpec
    metrics: dict[str, Any]
    final_learning_rate: list[float]


def _error(expected: str, provided: Any, path: str) -> DiagnosticInputError:
    return DiagnosticInputError(f"Expected {path} to be {expected}. Provided value: {provided!r}.")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise _error("a finite number", value, path)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error("a finite number", value, path) from exc
    if not math.isfinite(result):
        raise _error("a finite number", value, path)
    return result


def _history_final_learning_rate(path: Path, expected_length: int) -> list[float]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DiagnosticInputError(
            f"Expected canonical history.csv to be readable. Provided value: {path}."
        ) from exc
    if not rows:
        raise _error("a non-empty canonical history", rows, str(path))
    try:
        value = json.loads(rows[-1]["learning_rate"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise _error("JSON learning_rate in its final row", rows[-1], str(path)) from exc
    if not isinstance(value, list) or len(value) != expected_length:
        raise _error(f"a vector of exactly {expected_length} learning rates", value, str(path))
    return [_finite(item, f"{path}:final learning_rate") for item in value]


def _record(run_dir: Path, expected_run_id: str | None = None) -> BundleRecord:
    directory = run_dir.expanduser().resolve()
    manifest = validate_bundle(directory, expected_run_id)
    run_id = str(manifest["run_id"])
    spec = RunSpec.from_path(directory / "config.resolved.json")
    metrics = read_json(directory / "metrics.json")
    if not isinstance(metrics, dict):
        raise _error("a JSON object", metrics, str(directory / "metrics.json"))
    for key in ("best_test_accuracy", "final_test_accuracy"):
        _finite(metrics.get(key), f"metrics.{key}")
    depth = len(spec.data["run"]["architecture"]["channels"])
    final_lr = _history_final_learning_rate(directory / "history.csv", 2 * depth + 1)
    return BundleRecord(run_id, directory, spec, metrics, final_lr)


def _records_from_sweep(sweep_dir: Path) -> list[BundleRecord]:
    directory = sweep_dir.expanduser().resolve()
    manifest = load_manifest(directory / "manifest.json")
    layout = ResultLayout(directory.parent.parent)
    records = []
    for entry in manifest["entries"]:
        try:
            run_dir = layout.find_run_dir(entry["run_id"])
        except (RuntimeError, ValueError) as exc:
            raise DiagnosticInputError(
                "Expected each manifest run id to resolve to exactly one canonical bundle. "
                f"Provided run_id={entry['run_id']!r}, results_root={layout.root}."
            ) from exc
        if run_dir is None:
            raise DiagnosticInputError(
                f"Expected canonical run {entry['run_id']!r} from the sweep to be published. "
                f"Provided results root: {layout.root}."
            )
        records.append(_record(run_dir, entry["run_id"]))
    return records


def load_records(
    *,
    run_bundles: Iterable[str | Path] = (),
    sweep_dir: str | Path | None = None,
) -> list[BundleRecord]:
    paths = [Path(value) for value in run_bundles]
    if (sweep_dir is None) == (not paths):
        raise DiagnosticInputError(
            "Expected exactly one input mode: --sweep or one-or-more --run-bundle values. "
            f"Provided sweep={sweep_dir!r}, run_bundles={[str(path) for path in paths]!r}."
        )
    records = _records_from_sweep(Path(sweep_dir)) if sweep_dir is not None else [_record(path) for path in paths]
    seen: set[str] = set()
    for record in records:
        if record.run_id in seen:
            raise _error("unique canonical run ids", record.run_id, "diagnostic inputs")
        seen.add(record.run_id)
    return records


def _npz_parameters(path: Path) -> list[tuple[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "param_names" not in data:
                raise _error("param_names plus declared arrays", list(data.files), str(path))
            names = [str(item) for item in data["param_names"].tolist()]
            arrays = [(name, np.asarray(data[name])) for name in names]
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, DiagnosticInputError):
            raise
        raise DiagnosticInputError(
            f"Expected canonical NPZ weights to be readable. Provided value: {path}."
        ) from exc
    if not names or len(names) != len(set(names)):
        raise _error("a non-empty ordered list of unique parameter names", names, str(path))
    for name, array in arrays:
        if (
            array.size == 0
            or not np.issubdtype(array.dtype, np.number)
            or not np.isfinite(array).all()
        ):
            raise _error(
                "non-empty finite numeric parameter arrays",
                {name: {"dtype": str(array.dtype), "shape": list(array.shape)}},
                str(path),
            )
    return arrays


def _parameter_rows(record: BundleRecord, output_dir: Path) -> list[dict[str, Any]]:
    best = _npz_parameters(record.run_dir / "weights/best.npz")
    final = _npz_parameters(record.run_dir / "weights/final.npz")
    if [name for name, _ in best] != [name for name, _ in final]:
        raise _error("identical ordered best/final parameter names", [name for name, _ in final], str(record.run_dir))
    spec, run = record.spec.data, record.spec.data["run"]
    model, solver, dataset, training = run["model"], run["solver"], run["dataset"], run["training"]
    shared = {
        "run_id": record.run_id, "label": spec["label"],
        "protocol_id": spec["protocol_id"], "category": spec["category"],
        "architecture": run["architecture"]["profile"],
        "non_linearity": model["non_linearity"], "seed": spec["seed"],
        "voltage_amp": model["voltage_amp"], "current_amp": model["current_amp"],
        "input_gain": model["input_gain"],
        "inference_iterations": solver["inference_iterations"],
        "training_iterations": solver["training_iterations"],
        "batch_size": dataset["batch_size"], "epochs": training["epochs"],
        "run_bundle": relative_posix(record.run_dir, output_dir),
    }
    rows = []
    for (best_name, best_array), (final_name, final_array) in zip(best, final):
        if best_name != final_name or best_array.shape != final_array.shape:
            raise _error(
                "matching best/final parameter schemas",
                {"best": best_name, "final": final_name},
                str(record.run_dir),
            )
        best_flat = best_array.astype(np.float64, copy=False).reshape(-1)
        final_flat = final_array.astype(np.float64, copy=False).reshape(-1)
        delta = final_flat - best_flat
        best_l2 = float(np.linalg.norm(best_flat))
        final_l2 = float(np.linalg.norm(final_flat))
        delta_l2 = float(np.linalg.norm(delta))
        denominator = max(best_l2, final_l2, EPS)
        cosine_denominator = best_l2 * final_l2
        cosine = (
            float(np.dot(best_flat, final_flat) / cosine_denominator)
            if cosine_denominator > 0
            else ""
        )
        rows.append({
            **shared, "parameter": best_name, "element_count": int(best_flat.size),
            "best_l2": best_l2,
            "final_l2": final_l2, "delta_l2": delta_l2,
            "symmetric_relative_delta": delta_l2 / denominator, "cosine": cosine,
            "best_zero_fraction": float(np.mean(np.abs(best_flat) <= 1e-12)),
            "final_zero_fraction": float(np.mean(np.abs(final_flat) <= 1e-12)),
        })
    return rows


def analyze_records(
    records: Iterable[BundleRecord],
    output_dir: str | Path,
    *,
    required_nonlinearity: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    records = list(records)
    if not records:
        raise _error("at least one canonical run bundle", records, "diagnostic inputs")
    for record in records:
        if output == record.run_dir or output.is_relative_to(record.run_dir):
            raise DiagnosticInputError(
                "Expected --output-dir to be outside every immutable canonical run bundle. "
                f"Provided output={output}, run_bundle={record.run_dir}."
            )
    output.mkdir(parents=True, exist_ok=True)
    parameter_rows: list[dict[str, Any]] = []
    csv_summaries: list[dict[str, Any]] = []
    json_summaries: list[dict[str, Any]] = []
    for record in records:
        non_linearity = record.spec.data["run"]["model"]["non_linearity"]
        if required_nonlinearity is not None and non_linearity != required_nonlinearity:
            raise _error(required_nonlinearity, non_linearity, f"run {record.run_id} non_linearity")
        rows = _parameter_rows(record, output)
        parameter_rows.extend(rows)
        finite_cosines = [
            float(row["cosine"])
            for row in rows
            if row["cosine"] != "" and math.isfinite(float(row["cosine"]))
        ]
        relative = [float(row["symmetric_relative_delta"]) for row in rows]
        spec, run = record.spec.data, record.spec.data["run"]
        summary = {
            "run_id": record.run_id, "label": spec["label"],
            "protocol_id": spec["protocol_id"], "category": spec["category"],
            "architecture": run["architecture"]["profile"],
            "non_linearity": non_linearity, "seed": spec["seed"],
            "voltage_amp": run["model"]["voltage_amp"],
            "current_amp": run["model"]["current_amp"],
            "input_gain": run["model"]["input_gain"],
            "inference_iterations": run["solver"]["inference_iterations"],
            "training_iterations": run["solver"]["training_iterations"],
            "batch_size": run["dataset"]["batch_size"],
            "epochs": run["training"]["epochs"],
            "best_epoch": record.metrics["best_epoch"],
            "best_test_accuracy": record.metrics["best_test_accuracy"],
            "final_test_accuracy": record.metrics["final_test_accuracy"],
            "parameter_tensor_count": len(rows),
            "parameter_scalar_count": sum(int(row["element_count"]) for row in rows),
            "max_symmetric_relative_delta": max(relative),
            "mean_symmetric_relative_delta": sum(relative) / len(relative),
            "min_cosine": min(finite_cosines) if finite_cosines else None,
            "final_learning_rate": list(record.final_learning_rate),
            "run_bundle": relative_posix(record.run_dir, output),
        }
        json_summaries.append(summary)
        csv_summaries.append({
            **summary,
            "min_cosine": "" if summary["min_cosine"] is None else summary["min_cosine"],
            "final_learning_rate": json.dumps(
                summary["final_learning_rate"], separators=(",", ":")
            ),
        })
    atomic_write_csv(output / "checkpoint_update_ratios.csv", PARAMETER_COLUMNS, parameter_rows)
    atomic_write_csv(output / "summary.csv", SUMMARY_COLUMNS, csv_summaries)
    result = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "run_count": len(json_summaries),
        "parameter_rows": len(parameter_rows),
        "summary_path": "summary.csv",
        "parameter_path": "checkpoint_update_ratios.csv",
        "runs": json_summaries,
    }
    atomic_write_json(output / "diagnostic.json", result)
    return result


def _parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sweep", help="Canonical sweep directory whose manifest defines run order.")
    inputs.add_argument(
        "--run-bundle",
        action="append",
        help="Completed canonical run bundle; repeat for multiple runs.",
    )
    parser.add_argument("--output-dir", required=True, help="Destination for read-only diagnostic summaries.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    required_nonlinearity: str | None = None,
    description: str | None = None,
) -> int:
    parser = _parser(description)
    args = parser.parse_args(argv)
    try:
        records = load_records(run_bundles=args.run_bundle or (), sweep_dir=args.sweep)
        result = analyze_records(
            records, args.output_dir, required_nonlinearity=required_nonlinearity
        )
    except (DiagnosticInputError, InvalidBundleError, OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0
