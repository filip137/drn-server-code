#!/usr/bin/env python3
"""Analyze selected Conv1/Conv2 checkpoints with weights bounded to [1e-5,1e-4].

This is a read-only ordinary-MNIST mechanism diagnostic.  It combines the
canonical baseline/ours selections (including the terminal Conv2 continuation
choices) with the five completed historical legacy selections.  Historical
legacy coverage is kept in a separate evidence panel.

For every included surface the analyzer validates initialization,
best-validation, and final checkpoints; measures conductance distributions and
boundary transitions; imports recorded training-time projection diagnostics;
and replays raw gradients on the same deterministic training cohort without
applying optimizer steps.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LABS_ROOT = REPOSITORY_ROOT / "labs"
for import_root in (REPOSITORY_ROOT, LABS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.run_conv12_bounded_rho import build_source_config
from labs.datasets import stable_index_sequence_hash
from labs.mnist_train import (
    _resolve_callable,
    _resolve_dataset_config,
    load_config,
    train_mnist_conv,
)


SCHEMA = "perfectdiode-conv12-bounded-checkpoint-analysis/v1"
LOWER_CONDUCTANCE = 1.0e-5
UPPER_CONDUCTANCE = 1.0e-4
CONDUCTANCE_PREFIXES = ("ConvWeight_", "DenseWeight_")
BIAS_PREFIX = "Bias_"
ZERO_EPSILON = 1.0e-12

DEFAULT_ACTIVE_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "perfectdiode-conv12-bounded-rho-baseline-ours-tk46-seed0-v1"
)
DEFAULT_WAVE2_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1"
)
DEFAULT_WAVE3_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1"
)
DEFAULT_LEGACY_ROOT = Path(
    "/home/filip/server_code_conv_learning_rate_protocol/results/"
    "perfectdiode-conv12-initialization-rho-comparison-20260729-v4/"
    "attempt-generic-01"
)
DEFAULT_DATASET_ROOT = Path("/home/filip/datasets/mnist")
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "results"
    / "perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1"
    / "analysis"
)
FOCUSED_STUDY_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_bounded_rho_baseline_ours_20260729_v1.json"
)


LEGACY_STUDY_INITIALIZERS = {
    "lrinitstudy_73ed61a09bf317cf5452c670c63e9a85626fbd32603061c67052d1c0fab17876": (
        "bounded_uniform"
    ),
    "lrinitstudy_a3437c000f8397295ffb607dc18fc90cdc01313f043cdd8b4a08bc0ae935292a": (
        "bounded_kaiming_uniform"
    ),
}

# These are the five "Best completed" historical legacy cells in the validated
# predecessor summary.  Missing Conv2 legacy surfaces are intentionally not
# inferred or filled.
LEGACY_SELECTED_CELLS = (
    {
        "initializer": "bounded_uniform",
        "architecture": "conv1",
        "optimizer": "SGD",
        "study_id": "lrinitstudy_73ed61a09bf317cf5452c670c63e9a85626fbd32603061c67052d1c0fab17876",
        "cell_id": "pdcell_0c9c804308a42c9e01a51b9416f1a773ea9c8f8583076da890bbc3582560abed",
    },
    {
        "initializer": "bounded_uniform",
        "architecture": "conv1",
        "optimizer": "Adam",
        "study_id": "lrinitstudy_73ed61a09bf317cf5452c670c63e9a85626fbd32603061c67052d1c0fab17876",
        "cell_id": "pdcell_1d4da3c517b5b7048cd60d94c132b4e6cb3cf9d965e857a64b1b36212a1ae7d5",
    },
    {
        "initializer": "bounded_kaiming_uniform",
        "architecture": "conv1",
        "optimizer": "SGD",
        "study_id": "lrinitstudy_a3437c000f8397295ffb607dc18fc90cdc01313f043cdd8b4a08bc0ae935292a",
        "cell_id": "pdcell_2c74e202497325110c03652f1fb47461383853f2aa60c77a85153d61ff9a7371",
    },
    {
        "initializer": "bounded_kaiming_uniform",
        "architecture": "conv1",
        "optimizer": "Adam",
        "study_id": "lrinitstudy_a3437c000f8397295ffb607dc18fc90cdc01313f043cdd8b4a08bc0ae935292a",
        "cell_id": "pdcell_374f1066ccee6d723c38f99f0dd4c77f72ce75d680fa2324cfae7e02b9618b6f",
    },
    {
        "initializer": "bounded_uniform",
        "architecture": "conv2",
        "optimizer": "SGD",
        "study_id": "lrinitstudy_73ed61a09bf317cf5452c670c63e9a85626fbd32603061c67052d1c0fab17876",
        "cell_id": "pdcell_29a58428cc2f8722d02b8d4b1598762b9cb94e737fd7127fe339dd6c822d8207",
    },
)


@dataclass(frozen=True)
class Surface:
    surface_id: str
    evidence_panel: str
    architecture: str
    initializer: str
    scheme: str
    optimizer: str
    final_validation_accuracy: float
    final_validation_loss: float
    rho_conv: float
    rho_dense: float
    rho_range_classification: str
    rho_conv_edge: str | None
    rho_dense_edge: str | None
    cell_dir: Path
    initialization_checkpoint: Path
    best_checkpoint: Path
    final_checkpoint: Path
    best_epoch: int
    final_epoch: int
    config_path: Path | None
    run_spec_path: Path | None
    result_path: Path
    safety_path: Path | None
    step_log_path: Path | None
    selection_source_path: Path
    learning_rates_by_parameter: Mapping[str, float]
    source_policy_projection_is_report_only: bool

    def identity(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "evidence_panel": self.evidence_panel,
            "architecture": self.architecture,
            "initializer": self.initializer,
            "scheme": self.scheme,
            "optimizer": self.optimizer,
            "final_validation_accuracy": self.final_validation_accuracy,
            "final_validation_loss": self.final_validation_loss,
            "rho_conv": self.rho_conv,
            "rho_dense": self.rho_dense,
            "rho_range_classification": self.rho_range_classification,
            "rho_conv_edge": self.rho_conv_edge,
            "rho_dense_edge": self.rho_dense_edge,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Expected at least one row for {path}.")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {label}; found {value!r}.")
    return result


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    result = float(np.quantile(np.asarray(values, dtype=np.float64), q))
    return result if math.isfinite(result) else None


def _load_named_states(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "drn.function.parameters":
        raise ValueError(f"Unexpected checkpoint format in {path}.")
    schema = payload.get("schema")
    states = payload.get("states")
    if not isinstance(schema, list) or not isinstance(states, list):
        raise ValueError(f"Missing checkpoint schema or states in {path}.")
    if len(schema) != len(states):
        raise ValueError(f"Checkpoint schema/state mismatch in {path}.")
    result: dict[str, torch.Tensor] = {}
    for specification, state in zip(schema, states, strict=True):
        name = str(specification["name"]).strip()
        if name in result:
            raise ValueError(f"Duplicate parameter {name!r} in {path}.")
        result[name] = state.detach().cpu().contiguous()
    return result


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_set_hash(states: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(states.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_hash(value).encode("ascii"))
    return digest.hexdigest()


def _is_conductance(name: str) -> bool:
    return name.startswith(CONDUCTANCE_PREFIXES)


def _effective_bounds(dtype: torch.dtype) -> tuple[float, float]:
    lower = float(torch.tensor(LOWER_CONDUCTANCE, dtype=dtype).item())
    upper = float(torch.tensor(UPPER_CONDUCTANCE, dtype=dtype).item())
    return lower, upper


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = left.to(torch.float64).reshape(-1)
    y = right.to(torch.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 0.0:
        return None
    return float(torch.dot(x, y) / denominator)


def _distribution_stats(
    current: torch.Tensor,
    initial: torch.Tensor,
) -> dict[str, Any]:
    if current.shape != initial.shape:
        raise ValueError(
            f"Current/initial shape mismatch: {current.shape} != {initial.shape}."
        )
    lower, upper = _effective_bounds(current.dtype)
    width = upper - lower
    vector = current.to(torch.float64).reshape(-1)
    start = initial.to(torch.float64).reshape(-1)
    at_lower = vector == lower
    at_upper = vector == upper
    out_of_bounds = (vector < lower) | (vector > upper)
    if bool(out_of_bounds.any()):
        raise ValueError(
            f"Found {int(out_of_bounds.sum())} conductances outside [{lower},{upper}]."
        )
    delta = vector - start
    normalized = (vector - lower) / width
    result: dict[str, Any] = {
        "count": int(vector.numel()),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
        "mean": float(vector.mean()),
        "std": float(vector.std(unbiased=False)),
        "rms": float(torch.sqrt(torch.mean(vector.square()))),
        "normalized_mean_position": float(normalized.mean()),
        "exact_lower_fraction": float(at_lower.to(torch.float64).mean()),
        "exact_upper_fraction": float(at_upper.to(torch.float64).mean()),
        "combined_exact_bound_fraction": float(
            (at_lower | at_upper).to(torch.float64).mean()
        ),
        "near_lower_1pct_fraction": float(
            (normalized <= 0.01).to(torch.float64).mean()
        ),
        "near_upper_1pct_fraction": float(
            (normalized >= 0.99).to(torch.float64).mean()
        ),
        "near_lower_5pct_fraction": float(
            (normalized <= 0.05).to(torch.float64).mean()
        ),
        "near_upper_5pct_fraction": float(
            (normalized >= 0.95).to(torch.float64).mean()
        ),
        "mean_delta_from_initial": float(delta.mean()),
        "mean_abs_delta_from_initial": float(delta.abs().mean()),
        "rms_delta_from_initial": float(torch.sqrt(torch.mean(delta.square()))),
        "normalized_rms_delta_from_initial": float(
            torch.sqrt(torch.mean(delta.square())) / width
        ),
        "unchanged_from_initial_fraction": float(
            (delta == 0.0).to(torch.float64).mean()
        ),
        "moved_at_least_1pct_range_fraction": float(
            (delta.abs() >= 0.01 * width).to(torch.float64).mean()
        ),
        "moved_at_least_10pct_range_fraction": float(
            (delta.abs() >= 0.10 * width).to(torch.float64).mean()
        ),
        "initial_current_correlation": _correlation(start, vector),
    }
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        result[f"q{int(100 * q):02d}"] = float(torch.quantile(vector, q))
    return result


def _bias_stats(current: torch.Tensor, initial: torch.Tensor) -> dict[str, Any]:
    vector = current.to(torch.float64).reshape(-1)
    start = initial.to(torch.float64).reshape(-1)
    delta = vector - start
    return {
        "count": int(vector.numel()),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
        "mean": float(vector.mean()),
        "std": float(vector.std(unbiased=False)),
        "rms": float(torch.sqrt(torch.mean(vector.square()))),
        "mean_abs_delta_from_initial": float(delta.abs().mean()),
        "rms_delta_from_initial": float(torch.sqrt(torch.mean(delta.square()))),
        "unchanged_from_initial_fraction": float(
            (delta == 0.0).to(torch.float64).mean()
        ),
        "initial_current_correlation": _correlation(start, vector),
    }


def _bound_labels(value: torch.Tensor) -> torch.Tensor:
    lower, upper = _effective_bounds(value.dtype)
    vector = value.to(torch.float64).reshape(-1)
    labels = torch.ones(vector.numel(), dtype=torch.int8)
    labels[vector == lower] = 0
    labels[vector == upper] = 2
    return labels


def _transition_rows(
    surface: Surface,
    role: str,
    name: str,
    initial: torch.Tensor,
    current: torch.Tensor,
) -> list[dict[str, Any]]:
    labels = ("lower", "interior", "upper")
    source = _bound_labels(initial)
    target = _bound_labels(current)
    count = int(source.numel())
    rows = []
    for source_index, source_name in enumerate(labels):
        for target_index, target_name in enumerate(labels):
            transition_count = int(
                ((source == source_index) & (target == target_index)).sum()
            )
            rows.append(
                {
                    **surface.identity(),
                    "checkpoint_role": role,
                    "parameter_name": name,
                    "source_region": source_name,
                    "target_region": target_name,
                    "transition_count": transition_count,
                    "transition_fraction_all": transition_count / count,
                }
            )
    return rows


def _aggregate_states(
    states: Mapping[str, torch.Tensor], prefixes: tuple[str, ...]
) -> torch.Tensor:
    tensors = [
        value.reshape(-1)
        for name, value in sorted(states.items())
        if name.startswith(prefixes)
    ]
    if not tensors:
        raise ValueError(f"No states matched {prefixes!r}.")
    return torch.cat(tensors)


def _unit_norm_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    array = value.detach().cpu().to(torch.float64).numpy()
    if name.startswith("ConvWeight_"):
        unit_axis = "output_channel"
        reduction_axes = tuple(range(1, array.ndim))
    elif name.startswith("DenseWeight_"):
        unit_axis = "paired_output"
        reduction_axes = tuple(range(array.ndim - 1))
    else:
        raise ValueError(f"Unsupported unit-norm parameter {name!r}.")
    norms = np.sqrt(np.sum(array * array, axis=reduction_axes))
    energy = np.sort(norms * norms)[::-1]
    top_count = max(1, int(math.ceil(0.10 * len(norms))))
    mean = float(np.mean(norms))
    median = float(np.median(norms))
    return {
        "unit_axis": unit_axis,
        "unit_count": int(len(norms)),
        "zero_unit_count": int(np.sum(norms == 0.0)),
        "unit_norm_mean": mean,
        "unit_norm_std": float(np.std(norms)),
        "unit_norm_cv": float(np.std(norms) / mean) if mean > 0.0 else None,
        "unit_norm_median": median,
        "unit_norm_maximum": float(np.max(norms)),
        "maximum_over_median": (
            float(np.max(norms) / median) if median > 0.0 else None
        ),
        "top_10pct_unit_energy_fraction": float(
            np.sum(energy[:top_count]) / np.sum(energy)
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _surface_parts(surface_id: str) -> tuple[str, str, str, str]:
    parts = surface_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"Unexpected surface id {surface_id!r}.")
    initializer, architecture, scheme, optimizer = parts
    return initializer, architecture, scheme, optimizer.upper() if optimizer == "sgd" else "Adam"


def _find_active_cell(
    roots: Sequence[Path], surface_id: str, cell_id: str
) -> Path:
    candidates = [
        root / "surfaces" / surface_id / "rho" / "cells" / cell_id
        for root in roots
    ]
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"No local cell for {surface_id}/{cell_id}; tried {candidates!r}."
        )
    # Continuation roots are passed newest first.
    return matches[0].resolve()


def _artifact_expected_hash(result: Mapping[str, Any], name: str) -> str:
    matches = [
        record
        for record in result.get("artifacts", [])
        if Path(record.get("path", "")).name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one artifact record for {name}; found {matches!r}.")
    return str(matches[0]["sha256"])


def _discover_active_surfaces(
    active_root: Path,
    wave2_root: Path,
    wave3_root: Path,
) -> list[Surface]:
    active_summary_path = active_root / "summary.json"
    wave2_summary_path = wave2_root / "summary.json"
    wave3_summary_path = wave3_root / "summary.json"
    active_summary = _read_json(active_summary_path)
    wave2_summary = _read_json(wave2_summary_path)
    wave3_summary = _read_json(wave3_summary_path)
    if any(summary.get("official_test_read") is not False for summary in (active_summary, wave2_summary, wave3_summary)):
        raise ValueError("All active bounded summaries must record official_test_read=false.")

    selected_records: list[tuple[Mapping[str, Any], Path]] = []
    selected_records.extend(
        (record, active_summary_path)
        for record in active_summary["surfaces"]
        if record["architecture"] == "conv1"
    )
    conv2 = {
        record["surface_id"]: (record, wave2_summary_path)
        for record in wave2_summary["surfaces"]
    }
    conv2.update(
        {
            record["surface_id"]: (record, wave3_summary_path)
            for record in wave3_summary["surfaces"]
        }
    )
    selected_records.extend(conv2[surface_id] for surface_id in sorted(conv2))
    if len(selected_records) != 16:
        raise ValueError(
            f"Expected 16 canonical active surfaces; found {len(selected_records)}."
        )

    surfaces = []
    for record, selection_path in selected_records:
        surface_id = str(record["surface_id"])
        initializer, architecture, scheme, optimizer = _surface_parts(surface_id)
        selected = record["selected"]
        roots = (
            (wave3_root, wave2_root, active_root)
            if architecture == "conv2"
            else (active_root,)
        )
        cell_dir = _find_active_cell(roots, surface_id, str(selected["cell_id"]))
        result_path = cell_dir / "result.json"
        config_path = cell_dir / "config.used.json"
        cell_path = cell_dir / "cell.json"
        safety_path = cell_dir / "safety_diagnostics.json"
        for required in (result_path, config_path, cell_path, safety_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        result = _read_json(result_path)
        if result.get("completion", {}).get("official_test_read") is not False:
            raise ValueError(f"Selected run did not exclude official test: {cell_dir}.")
        cell = _read_json(cell_path)
        best_checkpoint = cell_dir / "best_model.pt"
        final_checkpoint = cell_dir / "final_model.pt"
        for name, path in (
            ("best_model.pt", best_checkpoint),
            ("final_model.pt", final_checkpoint),
        ):
            if _sha256_file(path) != _artifact_expected_hash(result, name):
                raise ValueError(f"Checkpoint artifact validation failed for {path}.")
        init_checkpoint = (
            active_root
            / "assets"
            / initializer
            / architecture
            / "final_model.pt"
        ).resolve()
        asset = _read_json(init_checkpoint.parent / "asset.json")
        if _sha256_file(init_checkpoint) != asset["checkpoint_sha256"]:
            raise ValueError(f"Initialization asset validation failed for {init_checkpoint}.")
        terminal = result["terminal_metrics"]
        range_status = record["rho_range_status"]
        surfaces.append(
            Surface(
                surface_id=surface_id,
                evidence_panel="active_matched_baseline_ours",
                architecture=architecture,
                initializer=initializer,
                scheme=scheme,
                optimizer=optimizer,
                final_validation_accuracy=_finite(
                    selected["final_validation_accuracy"], "validation accuracy"
                ),
                final_validation_loss=_finite(
                    selected["final_validation_loss"], "validation loss"
                ),
                rho_conv=_finite(selected["rho_conv"], "rho_conv"),
                rho_dense=_finite(selected["rho_dense"], "rho_dense"),
                rho_range_classification=str(range_status["classification"]),
                rho_conv_edge=range_status.get("rho_conv_edge"),
                rho_dense_edge=range_status.get("rho_dense_edge"),
                cell_dir=cell_dir,
                initialization_checkpoint=init_checkpoint,
                best_checkpoint=best_checkpoint.resolve(),
                final_checkpoint=final_checkpoint.resolve(),
                best_epoch=int(terminal["best_epoch"]),
                final_epoch=3,
                config_path=config_path.resolve(),
                run_spec_path=None,
                result_path=result_path.resolve(),
                safety_path=safety_path.resolve(),
                step_log_path=None,
                selection_source_path=selection_path.resolve(),
                learning_rates_by_parameter={
                    str(name): _finite(value, f"learning rate {name}")
                    for name, value in cell["learning_rates_by_parameter"].items()
                },
                source_policy_projection_is_report_only=True,
            )
        )
    return sorted(
        surfaces,
        key=lambda item: (
            item.architecture,
            item.initializer,
            item.scheme,
            item.optimizer,
        ),
    )


def _locate_legacy_cell(
    legacy_root: Path, study_id: str, cell_id: str
) -> Path:
    matches = sorted(
        legacy_root.glob(
            f"shards/*/studies/{study_id}/stages/rho_core_candidates/entries/{cell_id}"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one historical legacy cell for {study_id}/{cell_id}; "
            f"found {matches!r}."
        )
    return matches[0].resolve()


def _historical_edge(
    legacy_root: Path,
    architecture: str,
    optimizer: str,
    rho_conv: float,
    rho_dense: float,
) -> tuple[str | None, str | None]:
    config = _read_json(legacy_root / "provenance" / "config.json")
    row_id = f"{architecture}_legacy_v4_c0p25"
    matches = [
        surface
        for surface in config["surfaces"]
        if surface["row_id"] == row_id
        and str(surface["optimizer"]).lower() == optimizer.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing historical rho grid for {row_id}/{optimizer}.")
    match = matches[0]

    def edge(values: Sequence[float], selected: float) -> str | None:
        numbers = [float(value) for value in values]
        lower = math.isclose(selected, min(numbers), rel_tol=0.0, abs_tol=1e-15)
        upper = math.isclose(selected, max(numbers), rel_tol=0.0, abs_tol=1e-15)
        if lower and upper:
            return "both"
        if lower:
            return "lower"
        if upper:
            return "upper"
        return None

    return edge(match["rho_conv"], rho_conv), edge(match["rho_dense"], rho_dense)


def _discover_legacy_surfaces(legacy_root: Path) -> list[Surface]:
    surfaces = []
    for specification in LEGACY_SELECTED_CELLS:
        study_id = str(specification["study_id"])
        initializer = str(specification["initializer"])
        if LEGACY_STUDY_INITIALIZERS[study_id] != initializer:
            raise ValueError(f"Historical initializer/study mismatch: {specification}.")
        cell_dir = _locate_legacy_cell(
            legacy_root, study_id, str(specification["cell_id"])
        )
        result_path = cell_dir / "result.json"
        run_spec_path = cell_dir / "run_spec.json"
        step_log_path = cell_dir / "step_log.csv"
        result = _read_json(result_path)
        run_spec = _read_json(run_spec_path)
        if result.get("scheme") != "legacy" or not result.get("training_completed"):
            raise ValueError(f"Historical selected cell is not a completed legacy run: {cell_dir}.")
        if result.get("official_test_read") is not False:
            raise ValueError(f"Historical selected cell read the official test: {cell_dir}.")
        architecture = str(specification["architecture"])
        optimizer = str(specification["optimizer"])
        best_checkpoint = cell_dir / result["checkpoints"]["best_validation"]["path"]
        final_checkpoint = cell_dir / result["checkpoints"]["final"]["path"]
        for role, checkpoint in (
            ("best_validation", best_checkpoint),
            ("final", final_checkpoint),
        ):
            expected = result["checkpoints"][role]["sha256"]
            if _sha256_file(checkpoint) != expected:
                raise ValueError(f"Historical checkpoint validation failed: {checkpoint}.")
        study_root = cell_dir.parents[3]
        init_checkpoint = (
            study_root / "stages" / "assets" / "entries" / architecture / "initialization.pt"
        )
        initialization = _read_json(init_checkpoint.parent / "initialization.json")
        if _sha256_file(init_checkpoint) != initialization["checkpoint_sha256"]:
            raise ValueError(f"Historical initialization validation failed: {init_checkpoint}.")
        rho_conv = _finite(result["rho_conv"], "historical rho_conv")
        rho_dense = _finite(result["rho_dense"], "historical rho_dense")
        conv_edge, dense_edge = _historical_edge(
            legacy_root, architecture, optimizer, rho_conv, rho_dense
        )
        surfaces.append(
            Surface(
                surface_id=(
                    f"{initializer}__{architecture}__legacy__{optimizer.lower()}"
                ),
                evidence_panel="historical_legacy_predecessor",
                architecture=architecture,
                initializer=initializer,
                scheme="legacy",
                optimizer=optimizer,
                final_validation_accuracy=_finite(
                    result["final_validation_accuracy"], "historical validation accuracy"
                ),
                final_validation_loss=_finite(
                    result["final_validation_loss"], "historical validation loss"
                ),
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                rho_range_classification=(
                    "historical_grid_edge_selected"
                    if conv_edge is not None or dense_edge is not None
                    else "historical_grid_interior"
                ),
                rho_conv_edge=conv_edge,
                rho_dense_edge=dense_edge,
                cell_dir=cell_dir,
                initialization_checkpoint=init_checkpoint.resolve(),
                best_checkpoint=best_checkpoint.resolve(),
                final_checkpoint=final_checkpoint.resolve(),
                best_epoch=int(result["best_validation_epoch"]),
                final_epoch=int(result["epochs_completed"]),
                config_path=None,
                run_spec_path=run_spec_path.resolve(),
                result_path=result_path.resolve(),
                safety_path=None,
                step_log_path=step_log_path.resolve(),
                selection_source_path=(legacy_root / "_derived" / "run-summary.md").resolve(),
                learning_rates_by_parameter={
                    str(name): _finite(value, f"historical learning rate {name}")
                    for name, value in run_spec["raw_learning_rates_by_parameter"].items()
                },
                source_policy_projection_is_report_only=False,
            )
        )
    if len(surfaces) != 5:
        raise ValueError(f"Expected five historical legacy surfaces; found {len(surfaces)}.")
    return surfaces


def discover_surfaces(
    *,
    active_root: Path,
    wave2_root: Path,
    wave3_root: Path,
    legacy_root: Path,
) -> list[Surface]:
    surfaces = _discover_active_surfaces(active_root, wave2_root, wave3_root)
    surfaces.extend(_discover_legacy_surfaces(legacy_root))
    identifiers = [surface.surface_id for surface in surfaces]
    if len(set(identifiers)) != 21:
        raise ValueError(f"Expected 21 unique selected surfaces; found {identifiers!r}.")
    return sorted(
        surfaces,
        key=lambda item: (
            item.architecture,
            item.evidence_panel,
            item.initializer,
            item.scheme,
            item.optimizer,
        ),
    )


def _legacy_replay_config(surface: Surface) -> dict[str, Any]:
    study = _read_json(FOCUSED_STUDY_CONFIG)
    study = copy.deepcopy(study)
    study["model"]["schemes"]["legacy"] = {
        "voltage_amp": 4.0,
        "current_amp": 0.25,
    }
    config = build_source_config(
        study,
        initializer=surface.initializer,
        architecture=surface.architecture,
        scheme="legacy",
        optimizer=surface.optimizer,
        init_checkpoint_path=surface.initialization_checkpoint,
    )
    checkpoint_states = _load_named_states(surface.initialization_checkpoint)
    parameter_order = list(checkpoint_states)
    rates = [surface.learning_rates_by_parameter[name] for name in parameter_order]
    config["lr"] = rates
    config["optimizer"]["learning_rate"] = rates
    config["lab"]["epochs"] = 1
    config["analysis_provenance"] = {
        "evidence_panel": surface.evidence_panel,
        "historical_run_spec": str(surface.run_spec_path),
        "historical_result": str(surface.result_path),
        "source_policy_projection_is_report_only": False,
    }
    return config


def _surface_replay_config(surface: Surface, dataset_root: Path) -> dict[str, Any]:
    if surface.config_path is not None:
        config = _read_json(surface.config_path)
    else:
        config = _legacy_replay_config(surface)
    dataset_key = config["lab"]["dataset_key"]
    config["datasets"][dataset_key]["params"]["root"] = str(dataset_root.resolve())
    config["max_batches"] = None
    config["max_test_batches"] = None
    model = {
        **config["model_base"],
        **config["model_overrides"][config["lab"]["model_key"]],
    }
    if [float(model["weight_min"]), float(model["weight_max"])] != [
        LOWER_CONDUCTANCE,
        UPPER_CONDUCTANCE,
    ]:
        raise ValueError(f"Unexpected conductance bounds for {surface.surface_id}.")
    expected_tk = (4, 4) if surface.architecture == "conv1" else (6, 6)
    actual_tk = (
        int(model["num_iterations_inference"]),
        int(model["num_iterations_training"]),
    )
    if actual_tk != expected_tk:
        raise ValueError(
            f"Unexpected T/K for {surface.surface_id}: {actual_tk} != {expected_tk}."
        )
    return config


def _cohort_from_config(config_path: Path, replay_batches: int) -> dict[str, Any]:
    config = load_config(config_path)
    dataset_key, dataset_config = _resolve_dataset_config(
        config, config["lab"]["dataset_key"]
    )
    factory = _resolve_callable(dataset_config["factory"])
    params = dict(dataset_config["params"])
    params.setdefault("device", torch.device("cpu"))
    loaders = factory(**params).build()
    batches = loaders.train_batch_indices(num_epochs=1)[0][:replay_batches]
    if len(batches) != replay_batches:
        raise ValueError(
            f"Expected {replay_batches} deterministic replay batches; found {len(batches)}."
        )
    flattened = tuple(int(index) for batch in batches for index in batch)
    return {
        "dataset_key": dataset_key,
        "split_role": "ordinary_mnist_training_subset",
        "split_seed": int(loaders.split_seed),
        "shuffle_seed": int(loaders.shuffle_seed),
        "train_indices_sha256": loaders.train_indices_hash,
        "validation_indices_sha256": loaders.validation_indices_hash,
        "first_epoch_batch_order_sha256": loaders.first_epoch_batch_order_hash,
        "batch_count": replay_batches,
        "example_count": len(flattened),
        "source_indices": [list(map(int, batch)) for batch in batches],
        "source_indices_sha256": stable_index_sequence_hash(flattened),
        "batch_source_indices_sha256": [
            stable_index_sequence_hash(tuple(map(int, batch))) for batch in batches
        ],
    }


def _validate_cohort_provenance(
    surface: Surface, cohort: Mapping[str, Any]
) -> None:
    if surface.evidence_panel == "active_matched_baseline_ours":
        provenance = _read_json(surface.cell_dir / "metrics.json")[
            "dataset_provenance"
        ]
        keys = (
            "train_indices_sha256",
            "validation_indices_sha256",
            "first_epoch_batch_order_sha256",
        )
    else:
        provenance = _read_json(surface.result_path)
        keys = ("train_indices_sha256", "validation_indices_sha256")
    for key in keys:
        if provenance.get(key) != cohort.get(key):
            raise ValueError(
                f"Cohort provenance mismatch for {surface.surface_id}/{key}: "
                f"{provenance.get(key)!r} != {cohort.get(key)!r}."
            )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _vector_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.reshape(-1).to(torch.float64)))


def _gradient_geometry(
    *,
    state: torch.Tensor,
    gradient: torch.Tensor,
    name: str,
    learning_rate: float,
    optimizer: str,
    adam_epsilon: float,
) -> dict[str, Any]:
    vector = state.detach().cpu().reshape(-1).to(torch.float64)
    grad = gradient.detach().cpu().reshape(-1).to(torch.float64)
    grad_norm = _vector_norm(grad)
    result: dict[str, Any] = {
        "element_count": int(vector.numel()),
        "learning_rate": learning_rate,
        "gradient_l2": grad_norm,
        "gradient_rms": float(torch.sqrt(torch.mean(grad.square()))),
        "gradient_zero_fraction": float(
            (grad.abs() <= ZERO_EPSILON).to(torch.float64).mean()
        ),
    }
    if optimizer == "SGD":
        proposal = -learning_rate * grad
        proposal_kind = "exact_sgd_momentum0_weight_decay0"
    elif optimizer == "Adam":
        proposal = -learning_rate * grad / (grad.abs() + adam_epsilon)
        proposal_kind = "fresh_adam_shadow_without_saved_moments"
    else:
        raise ValueError(f"Unsupported optimizer {optimizer!r}.")
    result["proposal_kind"] = proposal_kind
    result["proposal_rms"] = float(torch.sqrt(torch.mean(proposal.square())))
    if not _is_conductance(name):
        result.update(
            {
                "exact_lower_fraction": None,
                "exact_upper_fraction": None,
                "outward_component_fraction_all": None,
                "outward_component_fraction_among_bound": None,
                "outward_gradient_energy_fraction": None,
                "tangent_gradient_efficiency_l2": None,
                "applied_update_rms": result["proposal_rms"],
                "projection_change_rms": 0.0,
                "projection_efficiency_l2": 1.0,
                "projection_changed_fraction": 0.0,
                "proposal_out_of_bounds_fraction": None,
            }
        )
        return result

    lower, upper = _effective_bounds(state.dtype)
    at_lower = vector == lower
    at_upper = vector == upper
    descent = -grad
    outward = (at_lower & (descent < 0.0)) | (at_upper & (descent > 0.0))
    tangent = descent.clone()
    tangent[outward] = 0.0
    tangent_norm = _vector_norm(tangent)
    outward_norm = _vector_norm(descent[outward]) if bool(outward.any()) else 0.0
    unprojected = vector + proposal
    projected = torch.clamp(unprojected, min=lower, max=upper)
    applied = projected - vector
    projection_change = projected - unprojected
    proposal_norm = _vector_norm(proposal)
    applied_norm = _vector_norm(applied)
    changed = projected != unprojected
    bound_count = int((at_lower | at_upper).sum())
    outward_count = int(outward.sum())
    result.update(
        {
            "exact_lower_fraction": float(at_lower.to(torch.float64).mean()),
            "exact_upper_fraction": float(at_upper.to(torch.float64).mean()),
            "outward_component_fraction_all": outward_count / int(vector.numel()),
            "outward_component_fraction_among_bound": (
                outward_count / bound_count if bound_count else None
            ),
            "outward_gradient_energy_fraction": (
                (outward_norm / grad_norm) ** 2 if grad_norm > 0.0 else None
            ),
            "tangent_gradient_efficiency_l2": _safe_ratio(
                tangent_norm, grad_norm
            ),
            "applied_update_rms": float(
                torch.sqrt(torch.mean(applied.square()))
            ),
            "projection_change_rms": float(
                torch.sqrt(torch.mean(projection_change.square()))
            ),
            "projection_efficiency_l2": _safe_ratio(
                applied_norm, proposal_norm
            ),
            "projection_changed_fraction": float(
                changed.to(torch.float64).mean()
            ),
            "proposal_out_of_bounds_fraction": float(
                ((unprojected < lower) | (unprojected > upper))
                .to(torch.float64)
                .mean()
            ),
        }
    )
    return result


def _gradient_comparison(
    current: torch.Tensor, initial: torch.Tensor
) -> dict[str, Any]:
    left = current.reshape(-1).to(torch.float64)
    right = initial.reshape(-1).to(torch.float64)
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    denominator = left_norm * right_norm
    cosine = (
        float(torch.dot(left, right) / denominator) if denominator > 0.0 else None
    )
    relative_delta = _safe_ratio(_vector_norm(left - right), right_norm)
    return {
        "gradient_l2_ratio_to_initial": _safe_ratio(left_norm, right_norm),
        "gradient_cosine_vs_initial": cosine,
        "gradient_relative_l2_delta_vs_initial": relative_delta,
    }


def _aggregate_gradient_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    group_keys = (
        "surface_id",
        "evidence_panel",
        "architecture",
        "initializer",
        "scheme",
        "optimizer",
        "checkpoint_role",
        "checkpoint_epoch",
        "parameter_name",
        "parameter_type",
        "proposal_kind",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)
    metrics = (
        "loss",
        "gradient_l2",
        "gradient_rms",
        "gradient_zero_fraction",
        "gradient_l2_ratio_to_initial",
        "gradient_cosine_vs_initial",
        "gradient_relative_l2_delta_vs_initial",
        "exact_lower_fraction",
        "exact_upper_fraction",
        "outward_component_fraction_all",
        "outward_component_fraction_among_bound",
        "outward_gradient_energy_fraction",
        "tangent_gradient_efficiency_l2",
        "proposal_rms",
        "applied_update_rms",
        "projection_change_rms",
        "projection_efficiency_l2",
        "projection_changed_fraction",
        "proposal_out_of_bounds_fraction",
    )
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary = {name: value for name, value in zip(group_keys, key, strict=True)}
        summary["batch_count"] = len(group)
        summary["element_count"] = int(group[0]["element_count"])
        summary["learning_rate"] = float(group[0]["learning_rate"])
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            summary[f"{metric}_median"] = _quantile(values, 0.50)
            summary[f"{metric}_q10"] = _quantile(values, 0.10)
            summary[f"{metric}_q90"] = _quantile(values, 0.90)
        output.append(summary)
    return output


def _checkpoint_roles(surface: Surface) -> tuple[tuple[str, int, Path], ...]:
    return (
        ("initial", 0, surface.initialization_checkpoint),
        ("best_validation", surface.best_epoch, surface.best_checkpoint),
        ("final", surface.final_epoch, surface.final_checkpoint),
    )


def _source_checkpoint_expected_hash(surface: Surface, role: str) -> str:
    if surface.evidence_panel == "active_matched_baseline_ours":
        if role == "initial":
            return str(_read_json(surface.initialization_checkpoint.parent / "asset.json")["checkpoint_sha256"])
        return _artifact_expected_hash(
            _read_json(surface.result_path),
            "best_model.pt" if role == "best_validation" else "final_model.pt",
        )
    if role == "initial":
        return str(_read_json(surface.initialization_checkpoint.parent / "initialization.json")["checkpoint_sha256"])
    result = _read_json(surface.result_path)
    key = "best_validation" if role == "best_validation" else "final"
    return str(result["checkpoints"][key]["sha256"])


def _replay_one_checkpoint(
    *,
    surface: Surface,
    checkpoint_role: str,
    checkpoint_epoch: int,
    checkpoint: Path,
    config: Mapping[str, Any],
    config_path: Path,
    replay_batches: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    before_file_hash = _sha256_file(checkpoint)
    expected_file_hash = _source_checkpoint_expected_hash(surface, checkpoint_role)
    if before_file_hash != expected_file_hash:
        raise ValueError(f"Source checkpoint hash mismatch for {checkpoint}.")
    before_states = _load_named_states(checkpoint)
    before_tensor_hash = _parameter_set_hash(before_states)
    learning_rates = {
        str(name): float(value)
        for name, value in surface.learning_rates_by_parameter.items()
    }
    if set(learning_rates) != set(before_states):
        raise ValueError(
            f"Learning-rate/parameter mismatch for {surface.surface_id}: "
            f"{set(learning_rates)} != {set(before_states)}."
        )
    optimizer = surface.optimizer
    adam_epsilon = float(config["optimizer"].get("eps", 1.0e-8))
    batch_rows: list[dict[str, Any]] = []
    gradients: dict[tuple[int, str], torch.Tensor] = {}

    def callback(payload: Mapping[str, Any]) -> bool:
        batch = int(payload["batch"])
        loss = _finite(payload["loss"], "replay loss")
        for parameter, gradient in zip(
            payload["parameters"], payload["gradients"], strict=True
        ):
            name = str(parameter.name).strip()
            gradient_cpu = gradient.detach().cpu().to(torch.float64).contiguous()
            gradients[(batch, name)] = gradient_cpu
            parameter_type = (
                "conductance_weight" if _is_conductance(name) else "bias"
            )
            batch_rows.append(
                {
                    **surface.identity(),
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint": str(checkpoint),
                    "batch": batch,
                    "loss": loss,
                    "parameter_name": name,
                    "parameter_type": parameter_type,
                    **_gradient_geometry(
                        state=parameter.state,
                        gradient=gradient,
                        name=name,
                        learning_rate=learning_rates[name],
                        optimizer=optimizer,
                        adam_epsilon=adam_epsilon,
                    ),
                }
            )
        return False

    with tempfile.TemporaryDirectory(prefix="bounded-rho-checkpoint-replay-") as temporary:
        replay_dir = Path(temporary) / "run"
        train_mnist_conv(
            config_path=config_path,
            epochs=1,
            lr=config["lr"],
            beta=config["beta"],
            log_interval=0,
            max_batches=replay_batches,
            max_test_batches=1,
            device=device,
            dataset_key=config["lab"]["dataset_key"],
            model_key=config["lab"]["model_key"],
            output_dir=replay_dir,
            training_algorithm=config["training_algorithm"],
            seed=config["seed"],
            lr_decay=config["optimizer"]["lr_decay"],
            init_checkpoint_path=checkpoint,
            batch_state_policy=config["batch_state_policy"],
            optimizer_name=optimizer,
            momentum=config["optimizer"].get("momentum"),
            weight_decay=config["optimizer"].get("weight_decay"),
            gradient_callback=callback,
            apply_optimizer_steps=False,
            skip_terminal_official_test=True,
        )
        replay_states = _load_named_states(replay_dir / "final_model.pt")
        if set(replay_states) != set(before_states) or any(
            not torch.equal(before_states[name], replay_states[name])
            for name in before_states
        ):
            raise RuntimeError(
                f"Read-only replay changed parameters for {surface.surface_id}/{checkpoint_role}."
            )

    after_file_hash = _sha256_file(checkpoint)
    after_states = _load_named_states(checkpoint)
    after_tensor_hash = _parameter_set_hash(after_states)
    if before_file_hash != after_file_hash or before_tensor_hash != after_tensor_hash:
        raise RuntimeError(f"Source checkpoint changed during replay: {checkpoint}.")
    expected_records = replay_batches * len(before_states)
    if len(batch_rows) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} gradient records; found {len(batch_rows)}."
        )
    guard = {
        **surface.identity(),
        "checkpoint_role": checkpoint_role,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint": str(checkpoint),
        "sha256_before": before_file_hash,
        "sha256_after": after_file_hash,
        "parameter_tensor_sha256_before": before_tensor_hash,
        "parameter_tensor_sha256_after": after_tensor_hash,
        "source_replay_tensor_match": True,
        "optimizer_steps_applied": False,
        "replay_reused": False,
    }
    return batch_rows, gradients, guard


def _replay_surface(
    *,
    surface: Surface,
    config: Mapping[str, Any],
    config_path: Path,
    replay_batches: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    gradients_by_role: dict[str, dict[tuple[int, str], torch.Tensor]] = {}
    guards: list[dict[str, Any]] = []
    cache: dict[str, tuple[str, list[dict[str, Any]], dict[tuple[int, str], torch.Tensor]]] = {}
    for role, epoch, checkpoint in _checkpoint_roles(surface):
        tensor_hash = _parameter_set_hash(_load_named_states(checkpoint))
        if tensor_hash in cache:
            source_role, cached_rows, cached_gradients = cache[tensor_hash]
            cloned_rows = [
                {
                    **row,
                    "checkpoint_role": role,
                    "checkpoint_epoch": epoch,
                    "checkpoint": str(checkpoint),
                }
                for row in cached_rows
            ]
            rows_by_role[role] = cloned_rows
            gradients_by_role[role] = {
                key: value.clone() for key, value in cached_gradients.items()
            }
            file_hash = _sha256_file(checkpoint)
            expected = _source_checkpoint_expected_hash(surface, role)
            if file_hash != expected:
                raise ValueError(f"Source checkpoint hash mismatch for {checkpoint}.")
            guards.append(
                {
                    **surface.identity(),
                    "checkpoint_role": role,
                    "checkpoint_epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "sha256_before": file_hash,
                    "sha256_after": file_hash,
                    "parameter_tensor_sha256_before": tensor_hash,
                    "parameter_tensor_sha256_after": tensor_hash,
                    "source_replay_tensor_match": True,
                    "optimizer_steps_applied": False,
                    "replay_reused": True,
                    "replay_reused_from_role": source_role,
                }
            )
            continue
        rows, gradients, guard = _replay_one_checkpoint(
            surface=surface,
            checkpoint_role=role,
            checkpoint_epoch=epoch,
            checkpoint=checkpoint,
            config=config,
            config_path=config_path,
            replay_batches=replay_batches,
            device=device,
        )
        rows_by_role[role] = rows
        gradients_by_role[role] = gradients
        guards.append(guard)
        cache[tensor_hash] = (role, rows, gradients)

    initial_gradients = gradients_by_role["initial"]
    all_rows = []
    for role, rows in rows_by_role.items():
        gradients = gradients_by_role[role]
        for row in rows:
            key = (int(row["batch"]), str(row["parameter_name"]))
            if key not in initial_gradients or key not in gradients:
                raise ValueError(f"Missing matched gradient vector {key}.")
            row.update(
                _gradient_comparison(gradients[key], initial_gradients[key])
            )
            all_rows.append(row)
    return all_rows, guards


def _training_projection_rows(surface: Surface) -> list[dict[str, Any]]:
    identity = surface.identity()
    analysis_policy = {
        "analysis_use": "report_only",
        "source_policy_projection_is_report_only": (
            surface.source_policy_projection_is_report_only
        ),
    }
    if surface.safety_path is not None:
        diagnostics = _read_json(surface.safety_path)
        names = sorted(
            set(diagnostics.get("initial_bound_occupancy_by_parameter", {}))
            | set(diagnostics.get("final_bound_occupancy_by_parameter", {}))
            | set(diagnostics.get("median_projection_efficiency_by_parameter", {}))
        )
        rows = []
        for name in names:
            rows.append(
                {
                    **identity,
                    **analysis_policy,
                    "parameter_name": name,
                    "record_scope": "parameter",
                    "processed_steps": diagnostics.get("processed_steps"),
                    "initial_bound_occupancy": diagnostics.get(
                        "initial_bound_occupancy_by_parameter", {}
                    ).get(name),
                    "final_bound_occupancy": diagnostics.get(
                        "final_bound_occupancy_by_parameter", {}
                    ).get(name),
                    "maximum_bound_occupancy": diagnostics.get(
                        "maximum_bound_occupancy_by_parameter", {}
                    ).get(name),
                    "median_projection_efficiency": diagnostics.get(
                        "median_projection_efficiency_by_parameter", {}
                    ).get(name),
                    "median_proposed_update_rms": diagnostics.get(
                        "median_proposed_update_rms_by_parameter", {}
                    ).get(name),
                    "median_applied_update_rms": diagnostics.get(
                        "median_applied_update_rms_by_parameter", {}
                    ).get(name),
                    "projection_efficiency_q10": None,
                    "projection_efficiency_q90": None,
                    "source": str(surface.safety_path),
                }
            )
        rows.append(
            {
                **identity,
                **analysis_policy,
                "parameter_name": "ALL_CONDUCTANCE_WEIGHTS",
                "record_scope": "aggregate",
                "processed_steps": diagnostics.get("processed_steps"),
                "initial_bound_occupancy": None,
                "final_bound_occupancy": None,
                "maximum_bound_occupancy": None,
                "median_projection_efficiency": diagnostics.get(
                    "median_projection_efficiency"
                ),
                "median_proposed_update_rms": None,
                "median_applied_update_rms": None,
                "projection_efficiency_q10": None,
                "projection_efficiency_q90": None,
                "source": str(surface.safety_path),
            }
        )
        return rows

    result = _read_json(surface.result_path)
    safety = result["safety"]
    step_values = []
    assert surface.step_log_path is not None
    with surface.step_log_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("median_projection_efficiency")
            if value not in (None, ""):
                step_values.append(_finite(value, "historical step projection"))
    names = sorted(safety.get("initial_bound_occupancy_by_parameter", {}))
    rows = [
        {
            **identity,
            **analysis_policy,
            "parameter_name": name,
            "record_scope": "parameter",
            "processed_steps": safety.get("processed_steps"),
            "initial_bound_occupancy": safety.get(
                "initial_bound_occupancy_by_parameter", {}
            ).get(name),
            "final_bound_occupancy": None,
            "maximum_bound_occupancy": safety.get(
                "maximum_bound_occupancy_by_parameter", {}
            ).get(name),
            "median_projection_efficiency": None,
            "median_proposed_update_rms": None,
            "median_applied_update_rms": None,
            "projection_efficiency_q10": None,
            "projection_efficiency_q90": None,
            "source": str(surface.result_path),
        }
        for name in names
    ]
    rows.append(
        {
            **identity,
            **analysis_policy,
            "parameter_name": "ALL_CONDUCTANCE_WEIGHTS",
            "record_scope": "aggregate",
            "processed_steps": safety.get("processed_steps"),
            "initial_bound_occupancy": None,
            "final_bound_occupancy": None,
            "maximum_bound_occupancy": None,
            "median_projection_efficiency": safety.get(
                "median_projection_efficiency"
            ),
            "median_proposed_update_rms": None,
            "median_applied_update_rms": None,
            "projection_efficiency_q10": _quantile(step_values, 0.10),
            "projection_efficiency_q90": _quantile(step_values, 0.90),
            "source": str(surface.step_log_path),
        }
    )
    return rows


def _weight_analysis(
    surfaces: Sequence[Surface],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, torch.Tensor]]],
]:
    weight_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    plot_states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for surface in surfaces:
        initial = _load_named_states(surface.initialization_checkpoint)
        roles = {
            "initial": initial,
            "best_validation": _load_named_states(surface.best_checkpoint),
            "final": _load_named_states(surface.final_checkpoint),
        }
        if any(set(states) != set(initial) for states in roles.values()):
            raise ValueError(f"Parameter mismatch within {surface.surface_id}.")
        plot_states[surface.surface_id] = roles
        epoch_by_role = {
            "initial": 0,
            "best_validation": surface.best_epoch,
            "final": surface.final_epoch,
        }
        for role, states in roles.items():
            common = {
                **surface.identity(),
                "checkpoint_role": role,
                "checkpoint_epoch": epoch_by_role[role],
            }
            for name, value in sorted(states.items()):
                if _is_conductance(name):
                    weight_rows.append(
                        {
                            **common,
                            "parameter_name": name,
                            **_distribution_stats(value, initial[name]),
                        }
                    )
                    transition_rows.extend(
                        _transition_rows(
                            surface, role, name, initial[name], value
                        )
                    )
                    unit_rows.append(
                        {
                            **common,
                            "parameter_name": name,
                            **_unit_norm_stats(value, name),
                        }
                    )
                elif name.startswith(BIAS_PREFIX):
                    bias_rows.append(
                        {
                            **common,
                            "parameter_name": name,
                            **_bias_stats(value, initial[name]),
                        }
                    )
            weight_rows.append(
                {
                    **common,
                    "parameter_name": "ALL_CONDUCTANCE_WEIGHTS",
                    **_distribution_stats(
                        _aggregate_states(states, CONDUCTANCE_PREFIXES),
                        _aggregate_states(initial, CONDUCTANCE_PREFIXES),
                    ),
                }
            )
    return weight_rows, bias_rows, transition_rows, unit_rows, plot_states


def _surface_summary_rows(
    surfaces: Sequence[Surface],
    weight_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for surface in surfaces:
        weights = [
            row
            for row in weight_rows
            if row["surface_id"] == surface.surface_id
            and row["checkpoint_role"] == "final"
        ]
        aggregate = next(
            row
            for row in weights
            if row["parameter_name"] == "ALL_CONDUCTANCE_WEIGHTS"
        )
        layers = [
            row
            for row in weights
            if row["parameter_name"] != "ALL_CONDUCTANCE_WEIGHTS"
        ]
        training = [
            row
            for row in training_rows
            if row["surface_id"] == surface.surface_id
        ]
        training_aggregate = next(
            row
            for row in training
            if row["parameter_name"] == "ALL_CONDUCTANCE_WEIGHTS"
        )
        training_layer_efficiencies = [
            float(row["median_projection_efficiency"])
            for row in training
            if row["parameter_name"] != "ALL_CONDUCTANCE_WEIGHTS"
            and row.get("median_projection_efficiency") is not None
        ]
        gradients = [
            row
            for row in gradient_rows
            if row["surface_id"] == surface.surface_id
            and row["checkpoint_role"] == "final"
            and row["parameter_type"] == "conductance_weight"
        ]
        tangent = [
            float(row["tangent_gradient_efficiency_l2_median"])
            for row in gradients
            if row.get("tangent_gradient_efficiency_l2_median") is not None
        ]
        proposal = [
            float(row["projection_efficiency_l2_median"])
            for row in gradients
            if row.get("projection_efficiency_l2_median") is not None
        ]
        zero = [
            float(row["gradient_zero_fraction_median"])
            for row in gradients
            if row.get("gradient_zero_fraction_median") is not None
        ]
        output.append(
            {
                **surface.identity(),
                "best_epoch": surface.best_epoch,
                "final_epoch": surface.final_epoch,
                "final_aggregate_exact_bound_fraction": aggregate[
                    "combined_exact_bound_fraction"
                ],
                "final_max_layer_exact_bound_fraction": max(
                    float(row["combined_exact_bound_fraction"]) for row in layers
                ),
                "final_aggregate_near_bound_1pct_fraction": (
                    float(aggregate["near_lower_1pct_fraction"])
                    + float(aggregate["near_upper_1pct_fraction"])
                ),
                "final_aggregate_normalized_rms_delta": aggregate[
                    "normalized_rms_delta_from_initial"
                ],
                "recorded_training_projection_efficiency": training_aggregate.get(
                    "median_projection_efficiency"
                ),
                "recorded_training_min_layer_projection_efficiency": (
                    min(training_layer_efficiencies)
                    if training_layer_efficiencies
                    else None
                ),
                "final_replay_min_tangent_gradient_efficiency": (
                    min(tangent) if tangent else None
                ),
                "final_replay_min_proposal_projection_efficiency": (
                    min(proposal) if proposal else None
                ),
                "final_replay_max_gradient_zero_fraction": max(zero) if zero else None,
                "proposal_interpretation": (
                    "exact_sgd"
                    if surface.optimizer == "SGD"
                    else "fresh_adam_shadow"
                ),
            }
        )
    return output


def _build_summary(
    surfaces: Sequence[Surface],
    surface_rows: Sequence[Mapping[str, Any]],
    weight_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    final_layers = [
        row
        for row in weight_rows
        if row["checkpoint_role"] == "final"
        and row["parameter_name"] != "ALL_CONDUCTANCE_WEIGHTS"
    ]
    final_aggregates = [
        row
        for row in weight_rows
        if row["checkpoint_role"] == "final"
        and row["parameter_name"] == "ALL_CONDUCTANCE_WEIGHTS"
    ]
    max_layer = max(
        final_layers, key=lambda row: float(row["combined_exact_bound_fraction"])
    )
    max_aggregate = max(
        final_aggregates,
        key=lambda row: float(row["combined_exact_bound_fraction"]),
    )
    active_training = [
        row
        for row in training_rows
        if row["evidence_panel"] == "active_matched_baseline_ours"
        and row["parameter_name"] != "ALL_CONDUCTANCE_WEIGHTS"
        and row.get("median_projection_efficiency") is not None
    ]
    min_training = min(
        active_training,
        key=lambda row: float(row["median_projection_efficiency"]),
    )
    final_gradients = [
        row
        for row in gradient_rows
        if row["checkpoint_role"] == "final"
        and row["parameter_type"] == "conductance_weight"
    ]
    tangent_rows = [
        row
        for row in final_gradients
        if row.get("tangent_gradient_efficiency_l2_median") is not None
    ]
    proposal_rows = [
        row
        for row in final_gradients
        if row.get("projection_efficiency_l2_median") is not None
    ]
    zero_rows = [
        row
        for row in final_gradients
        if float(row["gradient_zero_fraction_median"] or 0.0) >= 0.999999
    ]
    min_tangent = min(
        tangent_rows,
        key=lambda row: float(row["tangent_gradient_efficiency_l2_median"]),
    )
    sgd_proposals = [row for row in proposal_rows if row["optimizer"] == "SGD"]
    adam_proposals = [row for row in proposal_rows if row["optimizer"] == "Adam"]
    min_sgd = min(
        sgd_proposals,
        key=lambda row: float(row["projection_efficiency_l2_median"]),
    )
    min_adam = (
        min(
            adam_proposals,
            key=lambda row: float(row["projection_efficiency_l2_median"]),
        )
        if adam_proposals
        else None
    )
    return {
        "schema_version": SCHEMA,
        "generated_at": _utc_now(),
        "evidence_class": "ordinary_mnist_selection_mechanism_diagnostic",
        "official_test_read": False,
        "bounds": {"lower": LOWER_CONDUCTANCE, "upper": UPPER_CONDUCTANCE},
        "surface_count": len(surfaces),
        "active_matched_surface_count": sum(
            surface.evidence_panel == "active_matched_baseline_ours"
            for surface in surfaces
        ),
        "historical_legacy_surface_count": sum(
            surface.evidence_panel == "historical_legacy_predecessor"
            for surface in surfaces
        ),
        "checkpoint_roles": ["initial", "best_validation", "final"],
        "checkpoint_epochs_available": "initial/best/final only",
        "replay": {
            "device": device,
            "optimizer_steps_applied": False,
            "cohort": dict(cohort),
        },
        "extremes": {
            "maximum_final_layer_bound_occupancy": {
                "surface_id": max_layer["surface_id"],
                "parameter_name": max_layer["parameter_name"],
                "value": max_layer["combined_exact_bound_fraction"],
            },
            "maximum_final_aggregate_bound_occupancy": {
                "surface_id": max_aggregate["surface_id"],
                "value": max_aggregate["combined_exact_bound_fraction"],
            },
            "minimum_recorded_active_layer_projection_efficiency": {
                "surface_id": min_training["surface_id"],
                "parameter_name": min_training["parameter_name"],
                "value": min_training["median_projection_efficiency"],
            },
            "minimum_final_tangent_gradient_efficiency": {
                "surface_id": min_tangent["surface_id"],
                "parameter_name": min_tangent["parameter_name"],
                "value": min_tangent["tangent_gradient_efficiency_l2_median"],
            },
            "minimum_exact_sgd_proposal_projection_efficiency": {
                "surface_id": min_sgd["surface_id"],
                "parameter_name": min_sgd["parameter_name"],
                "value": min_sgd["projection_efficiency_l2_median"],
            },
            "minimum_fresh_adam_shadow_projection_efficiency": (
                {
                    "surface_id": min_adam["surface_id"],
                    "parameter_name": min_adam["parameter_name"],
                    "value": min_adam["projection_efficiency_l2_median"],
                }
                if min_adam is not None
                else None
            ),
        },
        "exact_zero_final_gradient_layers": [
            {
                "surface_id": row["surface_id"],
                "parameter_name": row["parameter_name"],
                "gradient_zero_fraction_median": row[
                    "gradient_zero_fraction_median"
                ],
            }
            for row in zero_rows
        ],
        "surface_summaries": list(surface_rows),
        "limitations": [
            "One model/loader seed and three training epochs per selected candidate.",
            "Ordinary-MNIST validation accuracy is diagnostic, not paper-facing evidence.",
            "Only initialization, best-validation, and final checkpoints exist; no intermediate epoch gradient curve is inferred.",
            "Historical legacy coverage is incomplete and came from a predecessor safety policy, so it is kept separate from the matched baseline/ours panel.",
            "Adam optimizer moments were not saved. Recorded training-time projection efficiency is authoritative; replayed Adam proposals are fresh-moment directional shadows.",
        ],
    }


def _surface_label(row: Mapping[str, Any]) -> str:
    initializer = "U" if row["initializer"] == "bounded_uniform" else "K"
    panel = "H" if row["evidence_panel"].startswith("historical") else "A"
    return (
        f"{panel}-{row['architecture']}-{initializer}-"
        f"{row['scheme']}-{row['optimizer']}"
    )


def _plot_weight_histograms(
    surfaces: Sequence[Surface],
    states: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    output_dir: Path,
) -> None:
    for architecture in ("conv1", "conv2"):
        selected = [surface for surface in surfaces if surface.architecture == architecture]
        if not selected:
            continue
        parameters = (
            ("ConvWeight_0", "DenseWeight_0")
            if architecture == "conv1"
            else ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
        )
        figure, axes = plt.subplots(
            len(selected),
            len(parameters),
            figsize=(4.0 * len(parameters), 2.1 * len(selected)),
            sharex=True,
            constrained_layout=True,
        )
        if len(selected) == 1:
            axes = np.asarray([axes])
        bins = np.linspace(0.0, 1.0, 41)
        for row_index, surface in enumerate(selected):
            for column, parameter in enumerate(parameters):
                axis = axes[row_index, column]
                for role, color, linewidth in (
                    ("initial", "0.35", 1.0),
                    ("final", "#2474b5" if surface.optimizer == "Adam" else "#d97620", 1.4),
                ):
                    value = states[surface.surface_id][role][parameter]
                    lower, upper = _effective_bounds(value.dtype)
                    normalized = (
                        value.to(torch.float64).reshape(-1).numpy() - lower
                    ) / (upper - lower)
                    axis.hist(
                        normalized,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=linewidth,
                        color=color,
                        label=role,
                    )
                axis.grid(alpha=0.2)
                if row_index == 0:
                    axis.set_title(parameter)
                if column == 0:
                    axis.set_ylabel(_surface_label(surface.identity()), fontsize=8)
                if row_index == len(selected) - 1:
                    axis.set_xlabel("normalized conductance position")
                if row_index == 0 and column == len(parameters) - 1:
                    axis.legend(fontsize=7)
        figure.suptitle(
            f"{architecture.capitalize()} selected bounded checkpoints: initialization vs final"
        )
        figure.savefig(output_dir / f"weight_histograms_{architecture}.png", dpi=180)
        plt.close(figure)


def _heatmap(
    *,
    rows: Sequence[Mapping[str, Any]],
    surfaces: Sequence[Surface],
    metrics: Sequence[tuple[str, str]],
    path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    parameters = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    labels = [_surface_label(surface.identity()) for surface in surfaces]
    figure, axes = plt.subplots(
        1, len(metrics), figsize=(5.0 * len(metrics), max(6.0, 0.38 * len(surfaces))), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    lookup = {
        (str(row["surface_id"]), str(row["parameter_name"])): row for row in rows
    }
    for axis, (metric, title) in zip(axes_array, metrics, strict=True):
        matrix = np.full((len(surfaces), len(parameters)), np.nan)
        for row_index, surface in enumerate(surfaces):
            for column, parameter in enumerate(parameters):
                record = lookup.get((surface.surface_id, parameter))
                if record is not None and record.get(metric) is not None:
                    matrix[row_index, column] = float(record[metric])
        image = axis.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        axis.set_xticks(range(len(parameters)), parameters, rotation=20)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
        axis.set_title(title)
        for row_index in range(len(surfaces)):
            for column in range(len(parameters)):
                value = matrix[row_index, column]
                axis.text(
                    column,
                    row_index,
                    "—" if not math.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if math.isfinite(value) and value < 0.45 else "black",
                )
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_accuracy_mechanisms(
    surface_rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    fields = (
        ("final_aggregate_exact_bound_fraction", "final exact-bound occupancy"),
        ("recorded_training_projection_efficiency", "recorded training projection efficiency"),
        ("final_replay_min_tangent_gradient_efficiency", "final feasible-gradient L2 fraction"),
    )
    colors = {"baseline": "#4c78a8", "ours": "#e45756", "legacy": "#54a24b"}
    markers = {"SGD": "o", "Adam": "s"}
    for axis, (field, title) in zip(axes, fields, strict=True):
        for row in surface_rows:
            value = row.get(field)
            if value is None:
                continue
            axis.scatter(
                float(value),
                100.0 * float(row["final_validation_accuracy"]),
                c=colors[str(row["scheme"])],
                marker=markers[str(row["optimizer"])],
                alpha=0.85,
                edgecolors="black",
                linewidths=0.3,
            )
        axis.set_xlabel(title)
        axis.set_ylabel("ordinary-MNIST validation accuracy (%)")
        axis.grid(alpha=0.25)
    figure.suptitle("Mechanism diagnostics versus three-epoch selection accuracy")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    surface_rows: Sequence[Mapping[str, Any]],
) -> None:
    extremes = summary["extremes"]
    adam_extreme = extremes["minimum_fresh_adam_shadow_projection_efficiency"]
    adam_extreme_line = (
        "- Fresh-moment Adam shadow projection efficiency: not covered by this "
        "subset."
        if adam_extreme is None
        else (
            "- Lowest fresh-moment Adam shadow projection efficiency: "
            f"{float(adam_extreme['value']):.3f}; this is not a reconstruction "
            "of unsaved Adam moments."
        )
    )
    lines = [
        "# Bounded Conv1/Conv2 checkpoint mechanism analysis",
        "",
        "This is a one-seed, three-epoch ordinary-MNIST diagnostic. It is not paper-facing accuracy evidence.",
        "",
        "## Coverage",
        "",
        f"- {summary['active_matched_surface_count']} matched baseline/ours surfaces.",
        f"- {summary['historical_legacy_surface_count']} completed historical legacy surfaces, reported separately.",
        "- Only initialization, best-validation, and final checkpoints are available; no intermediate epochs are inferred.",
        f"- Replay cohort: {summary['replay']['cohort']['example_count']} training examples in {summary['replay']['cohort']['batch_count']} fixed batches, hash `{summary['replay']['cohort']['source_indices_sha256']}`.",
        "- No optimizer steps were applied during replay; every source checkpoint passed before/after byte and tensor guards.",
        "",
        "## Measured extremes",
        "",
        f"- Maximum final layer exact-bound occupancy: {100.0 * float(extremes['maximum_final_layer_bound_occupancy']['value']):.2f}% in `{extremes['maximum_final_layer_bound_occupancy']['surface_id']}` / `{extremes['maximum_final_layer_bound_occupancy']['parameter_name']}`.",
        f"- Maximum aggregate final exact-bound occupancy: {100.0 * float(extremes['maximum_final_aggregate_bound_occupancy']['value']):.2f}% in `{extremes['maximum_final_aggregate_bound_occupancy']['surface_id']}`.",
        f"- Lowest recorded active per-layer training projection efficiency: {float(extremes['minimum_recorded_active_layer_projection_efficiency']['value']):.3f} in `{extremes['minimum_recorded_active_layer_projection_efficiency']['surface_id']}` / `{extremes['minimum_recorded_active_layer_projection_efficiency']['parameter_name']}`.",
        f"- Lowest final feasible raw-gradient L2 fraction: {float(extremes['minimum_final_tangent_gradient_efficiency']['value']):.3f} in `{extremes['minimum_final_tangent_gradient_efficiency']['surface_id']}` / `{extremes['minimum_final_tangent_gradient_efficiency']['parameter_name']}`.",
        f"- Lowest exact SGD one-step projection efficiency: {float(extremes['minimum_exact_sgd_proposal_projection_efficiency']['value']):.3f}.",
        adam_extreme_line,
        "",
        "## Surface summary",
        "",
        "| Surface | Accuracy | rho conv/dense | rho range | Aggregate bound | Training projection | Final tangent efficiency | Final proposal efficiency |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in surface_rows:
        training = row.get("recorded_training_projection_efficiency")
        tangent = row.get("final_replay_min_tangent_gradient_efficiency")
        proposal = row.get("final_replay_min_proposal_projection_efficiency")
        lines.append(
            "| "
            + f"`{row['surface_id']}` | {100.0 * float(row['final_validation_accuracy']):.2f}% "
            + f"| {float(row['rho_conv']):.6g}/{float(row['rho_dense']):.6g} "
            + f"| {row['rho_range_classification']} "
            + f"| {100.0 * float(row['final_aggregate_exact_bound_fraction']):.1f}% "
            + f"| {'—' if training is None else f'{float(training):.3f}'} "
            + f"| {'—' if tangent is None else f'{float(tangent):.3f}'} "
            + f"| {'—' if proposal is None else f'{float(proposal):.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guards",
            "",
            "- Bound occupancy and projection efficiency are report-only in this analysis and do not exclude a checkpoint.",
            "- Historical legacy used an older safety policy and lacks complete Conv2 initializer/optimizer coverage.",
            "- SGD proposal replay is exact for momentum and weight decay zero. Adam proposal replay is a fresh-state directional shadow; recorded training diagnostics carry more evidential weight.",
            "- Weight distributions and checkpoint replay establish association, not causality.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def run_analysis(args: argparse.Namespace) -> Path:
    surfaces = discover_surfaces(
        active_root=args.active_root.resolve(),
        wave2_root=args.wave2_root.resolve(),
        wave3_root=args.wave3_root.resolve(),
        legacy_root=args.legacy_root.resolve(),
    )
    if args.surface_id:
        requested = set(args.surface_id)
        surfaces = [surface for surface in surfaces if surface.surface_id in requested]
        missing = requested - {surface.surface_id for surface in surfaces}
        if missing:
            raise ValueError(f"Unknown requested surfaces: {sorted(missing)!r}.")
    if args.smoke:
        smoke_ids = {
            "bounded_uniform__conv1__baseline__sgd",
            "bounded_uniform__conv2__ours__sgd",
            "bounded_uniform__conv1__legacy__sgd",
        }
        surfaces = [surface for surface in surfaces if surface.surface_id in smoke_ids]
        args.replay_batches = 1
    if not surfaces:
        raise ValueError("No surfaces selected for analysis.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    replay_config_dir = output_dir / "replay_configs"
    replay_config_dir.mkdir()
    inventory_rows = []
    source_result_records = []
    replay_configs: dict[str, tuple[dict[str, Any], Path]] = {}
    cohort: dict[str, Any] | None = None
    for surface in surfaces:
        config = _surface_replay_config(surface, args.dataset_root.resolve())
        config_path = replay_config_dir / f"{surface.surface_id}.json"
        _write_json(config_path, config)
        current_cohort = _cohort_from_config(config_path, args.replay_batches)
        if cohort is None:
            cohort = current_cohort
        elif current_cohort != cohort:
            raise ValueError(f"Replay cohort differs for {surface.surface_id}.")
        _validate_cohort_provenance(surface, current_cohort)
        replay_configs[surface.surface_id] = (config, config_path)
        source_result_records.append(
            {
                "surface_id": surface.surface_id,
                "result_path": str(surface.result_path),
                "result_sha256": _sha256_file(surface.result_path),
                "selection_source_path": str(surface.selection_source_path),
                "selection_source_sha256": _sha256_file(surface.selection_source_path),
            }
        )
        for role, epoch, checkpoint in _checkpoint_roles(surface):
            states = _load_named_states(checkpoint)
            actual = _sha256_file(checkpoint)
            expected = _source_checkpoint_expected_hash(surface, role)
            if actual != expected:
                raise ValueError(f"Checkpoint inventory hash mismatch: {checkpoint}.")
            inventory_rows.append(
                {
                    **surface.identity(),
                    "checkpoint_role": role,
                    "checkpoint_epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": actual,
                    "expected_sha256": expected,
                    "parameter_tensor_sha256": _parameter_set_hash(states),
                    "parameter_names": ";".join(states),
                    "replay_config": str(config_path),
                    "replay_config_sha256": _sha256_file(config_path),
                }
            )
    assert cohort is not None

    weight_rows, bias_rows, transition_rows, unit_rows, plot_states = _weight_analysis(
        surfaces
    )
    training_rows = [
        row for surface in surfaces for row in _training_projection_rows(surface)
    ]
    gradient_batch_rows: list[dict[str, Any]] = []
    guards = []
    for index, surface in enumerate(surfaces, start=1):
        print(
            f"[{index}/{len(surfaces)}] replay {surface.surface_id}",
            flush=True,
        )
        config, config_path = replay_configs[surface.surface_id]
        rows, surface_guards = _replay_surface(
            surface=surface,
            config=config,
            config_path=config_path,
            replay_batches=args.replay_batches,
            device=args.device,
        )
        gradient_batch_rows.extend(rows)
        guards.extend(surface_guards)
    gradient_summary_rows = _aggregate_gradient_rows(gradient_batch_rows)
    surface_rows = _surface_summary_rows(
        surfaces, weight_rows, training_rows, gradient_summary_rows
    )
    summary = _build_summary(
        surfaces,
        surface_rows,
        weight_rows,
        training_rows,
        gradient_summary_rows,
        cohort,
        args.device,
    )

    _write_csv(output_dir / "checkpoint_inventory.csv", inventory_rows)
    _write_csv(output_dir / "weight_distribution_by_layer.csv", weight_rows)
    _write_csv(output_dir / "bias_distribution_by_layer.csv", bias_rows)
    _write_csv(output_dir / "weight_boundary_transitions.csv", transition_rows)
    _write_csv(output_dir / "weight_unit_norm_summary.csv", unit_rows)
    _write_csv(output_dir / "training_projection_summary.csv", training_rows)
    _write_csv(output_dir / "checkpoint_gradient_by_batch.csv", gradient_batch_rows)
    _write_csv(
        output_dir / "checkpoint_gradient_summary.csv", gradient_summary_rows
    )
    _write_csv(output_dir / "surface_summary.csv", surface_rows)
    _write_json(output_dir / "checkpoint_guards.json", guards)
    _write_json(output_dir / "replay_cohort.json", cohort)
    _write_json(output_dir / "source_results.json", source_result_records)
    _write_json(output_dir / "summary.json", summary)

    final_weight_layers = [
        row
        for row in weight_rows
        if row["checkpoint_role"] == "final"
        and row["parameter_name"] != "ALL_CONDUCTANCE_WEIGHTS"
    ]
    _plot_weight_histograms(surfaces, plot_states, output_dir)
    _heatmap(
        rows=final_weight_layers,
        surfaces=surfaces,
        metrics=(("combined_exact_bound_fraction", "final exact-bound occupancy"),),
        path=output_dir / "final_bound_occupancy.png",
    )
    final_gradient_layers = [
        row
        for row in gradient_summary_rows
        if row["checkpoint_role"] == "final"
        and row["parameter_type"] == "conductance_weight"
    ]
    _heatmap(
        rows=final_gradient_layers,
        surfaces=surfaces,
        metrics=(
            ("tangent_gradient_efficiency_l2_median", "feasible raw-gradient L2 fraction"),
            ("projection_efficiency_l2_median", "one-step proposal efficiency"),
            ("gradient_cosine_vs_initial_median", "gradient cosine vs initialization"),
        ),
        path=output_dir / "final_gradient_projection.png",
    )
    _plot_accuracy_mechanisms(
        surface_rows, output_dir / "accuracy_mechanism_scatter.png"
    )
    _write_report(output_dir / "report.md", summary, surface_rows)

    git = _git_state()
    environment = {
        "generated_at": _utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    _write_json(output_dir / "environment.json", environment)
    _write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": SCHEMA,
            "created_at": _utc_now(),
            "study_id": "perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1",
            "evidence_class": "ordinary_mnist_selection_mechanism_diagnostic",
            "official_test_read": False,
            "command": [sys.executable, *sys.argv],
            "git": git,
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "input_roots": {
                "active": str(args.active_root.resolve()),
                "wave2": str(args.wave2_root.resolve()),
                "wave3": str(args.wave3_root.resolve()),
                "historical_legacy": str(args.legacy_root.resolve()),
                "dataset": str(args.dataset_root.resolve()),
            },
            "output_dir": str(output_dir),
            "surface_count": len(surfaces),
            "replay_batches": args.replay_batches,
            "replay_examples": cohort["example_count"],
            "device": args.device,
            "smoke": bool(args.smoke),
            "optimizer_steps_applied": False,
            "source_result_count": len(source_result_records),
            "source_result_manifest_sha256": _sha256_json(source_result_records),
            "checkpoint_guard_count": len(guards),
            "checkpoint_guards_passed": all(
                guard["sha256_before"] == guard["sha256_after"]
                and guard["parameter_tensor_sha256_before"]
                == guard["parameter_tensor_sha256_after"]
                and guard["source_replay_tensor_match"]
                and not guard["optimizer_steps_applied"]
                for guard in guards
            ),
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, default=DEFAULT_ACTIVE_ROOT)
    parser.add_argument("--wave2-root", type=Path, default=DEFAULT_WAVE2_ROOT)
    parser.add_argument("--wave3-root", type=Path, default=DEFAULT_WAVE3_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-batches", type=int, default=16)
    parser.add_argument("--surface-id", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.replay_batches <= 0:
        raise ValueError("--replay-batches must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Analysis output already exists; use a new path: {args.output_dir}."
        )
    output = run_analysis(args)
    print(output, flush=True)


if __name__ == "__main__":
    main()
