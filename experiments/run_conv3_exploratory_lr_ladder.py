#!/usr/bin/env python3
"""Run the no-canary Conv3 exploratory learning-rate ladder.

This runner is deliberately separate from the formal bounded-rho selector.  It
uses that selector's pinned Conv3 initializer and source-config construction,
but executes every cell in an explicitly exploratory 8x8 Cartesian grid.  The
grid grows outwards from its lowest/lowest corner and has no accuracy, safety,
or post-training T/K rejection gate.  Non-finite training values and ordinary
runtime failures remain terminal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rho_search import (
    _float_token,
    _git_state,
    _write_json,
    run as run_rho_search,
)
from experiments.run_conv12_bounded_rho import (
    build_source_config,
    ensure_asset,
    load_study as load_parent_study,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_exploratory_lr_ladder_seed0_20260811_v1.json"
)
STUDY_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-study/v1"
RESOLVED_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-resolved/v1"
SURFACE_STATUS_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-status/v1"
SURFACE_PROGRESS_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-progress/v1"
SURFACE_SUMMARY_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-surface/v1"
STUDY_SUMMARY_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-summary/v1"
SMOKE_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-smoke/v1"
EXPECTED_PARENT_SCHEMA = (
    "perfectdiode-conv3-bounded-uniform-zero-bias-rho-study/v1"
)
EXPECTED_PARENT_STUDY_ID = (
    "perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-"
    "occupancy-report-only-seed0-20260811-v1"
)
EXPECTED_PARENT_SHA256 = (
    "b845340aec78237dd0f6940a6dfcec76a4463bee0fe7164ae6ea51f5bc68d647"
)
EXPECTED_INITIALIZER_SHA256 = (
    "8ebe9dd916e1e9299c31053e2bb37b4f5121f8322f26af7746a92875e555438f"
)
TERMINAL_CELL_STATUSES = {"complete", "candidate_rejected_nonfinite"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_axes() -> tuple[list[float], list[float]]:
    return (
        [0.009 / (3.0**power) for power in range(7, -1, -1)],
        [0.03 / (3.0**power) for power in range(7, -1, -1)],
    )


def _require_exact_float_list(
    observed: Any, expected: Sequence[float], *, label: str
) -> list[float]:
    if not isinstance(observed, list):
        raise ValueError(f"Expected {label} to be a list.")
    values = [float(value) for value in observed]
    if len(values) != len(expected) or any(
        not math.isclose(left, right, rel_tol=1e-15, abs_tol=0.0)
        for left, right in zip(values, expected)
    ):
        raise ValueError(
            f"Expected {label}={list(expected)!r}. Provided value: {values!r}."
        )
    return values


def load_study(
    path: str | Path = DEFAULT_STUDY,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Load and fail-closed validate the exploratory and parent contracts."""

    source = Path(path).expanduser().resolve()
    study = json.loads(source.read_text(encoding="utf-8"))
    if study.get("schema_version") != STUDY_SCHEMA:
        raise ValueError(f"Unexpected exploratory study schema: {study.get('schema_version')!r}.")
    if study.get("evidence_class") != "ordinary_mnist_exploratory":
        raise ValueError("The LR ladder must be ordinary-MNIST exploratory evidence.")
    expected_scope = {
        "architectures": ["conv3"],
        "schemes": ["baseline", "ours"],
        "optimizers": ["SGD", "Adam"],
        "initializers": ["bounded_uniform"],
        "excluded": ["legacy", "conv1", "conv2"],
    }
    if study.get("scope") != expected_scope:
        raise ValueError(f"Expected exploratory scope {expected_scope!r}.")

    parent_record = study.get("parent")
    if not isinstance(parent_record, Mapping):
        raise ValueError("The exploratory study requires a parent record.")
    parent_relative = parent_record.get("study_config")
    if not isinstance(parent_relative, str) or not parent_relative:
        raise ValueError("The exploratory parent study_config must be a path string.")
    parent_path = (REPO_ROOT / parent_relative).resolve()
    if not parent_path.is_relative_to(REPO_ROOT):
        raise ValueError("The exploratory parent config must stay beneath the repository.")
    if parent_record.get("study_id") != EXPECTED_PARENT_STUDY_ID:
        raise ValueError("Unexpected exploratory parent study identity.")
    if parent_record.get("study_config_sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("Unexpected pinned parent-config SHA-256.")
    if _sha256_file(parent_path) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("The current Conv3 parent config does not match its pinned SHA-256.")
    loaded_parent_path, parent = load_parent_study(parent_path)
    if loaded_parent_path != parent_path:
        raise RuntimeError("The parent loader resolved a different config path.")
    if parent.get("schema_version") != EXPECTED_PARENT_SCHEMA:
        raise ValueError("Unexpected parent study schema.")
    if parent.get("study_id") != EXPECTED_PARENT_STUDY_ID:
        raise ValueError("Unexpected loaded parent study identity.")
    expected_reference = {
        "source_study_id": (
            "perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-"
            "seed0-20260810-v1"
        ),
        "checkpoint_sha256_by_architecture": {
            "conv3": EXPECTED_INITIALIZER_SHA256,
        },
    }
    if parent.get("initialization_reference") != expected_reference:
        raise ValueError("The parent does not pin the required Conv3 initializer.")
    if parent_record.get("initializer_checkpoint_sha256") != EXPECTED_INITIALIZER_SHA256:
        raise ValueError("Unexpected exploratory initializer checkpoint SHA-256.")

    model = parent["model"]
    if [float(model["weight_min"]), float(model["weight_max"])] != [1e-5, 1e-4]:
        raise ValueError("The parent weight interval is not [1e-5,1e-4].")
    architecture = model["architectures"]["conv3"]
    if [int(architecture["T"]), int(architecture["K"])] != [8, 8]:
        raise ValueError("The parent Conv3 operating point is not T=K=8.")
    if parent.get("bias_contract") != {
        "initialization": "default_zero",
        "learning_rate": 0.0,
        "conductance_projection": False,
    }:
        raise ValueError("The parent does not retain the exact zero-bias contract.")
    if parent["dataset"].get("official_test_read") is not False:
        raise ValueError("The parent must leave the official test split unread.")

    search = study.get("rho_search")
    if not isinstance(search, Mapping):
        raise ValueError("The exploratory study requires rho_search.")
    expected_conv, expected_dense = _expected_axes()
    _require_exact_float_list(search.get("rho_conv"), expected_conv, label="rho_conv")
    _require_exact_float_list(search.get("rho_dense"), expected_dense, label="rho_dense")
    required_search = {
        "cell_order": [
            "max_axis_index",
            "sum_axis_indices",
            "conv_axis_index",
            "dense_axis_index",
        ],
        "probe_batches": [128],
        "probe_stability_tolerance": 0.5,
        "bias_policy": "zero",
        "candidate_epochs": 3,
        "expected_candidate_steps": 10314,
        "minimum_validation_accuracy": 0.0,
        "skip_canary": True,
        "canary_steps": 0,
        "disable_safety_rejections": True,
        "post_candidate_tk": False,
    }
    for key, expected in required_search.items():
        if search.get(key) != expected:
            raise ValueError(
                f"Expected rho_search.{key}={expected!r}, got {search.get(key)!r}."
            )
    safety = search.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("The exploratory study requires explicit safety diagnostics.")
    if safety.get("bound_occupancy_increase_maximum") is not None:
        raise ValueError("Bound occupancy must remain diagnostic-only.")
    if safety.get("projection_efficiency_minimum") is not None:
        raise ValueError("Projection efficiency must remain diagnostic-only.")
    return source, study, parent_path, parent


def surface_specs(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for initializer in study["scope"]["initializers"]:
        for architecture in study["scope"]["architectures"]:
            for scheme in study["scope"]["schemes"]:
                for optimizer in study["scope"]["optimizers"]:
                    rows.append(
                        {
                            "index": len(rows),
                            "initializer": initializer,
                            "architecture": architecture,
                            "scheme": scheme,
                            "optimizer": optimizer,
                            "surface_id": (
                                f"{initializer}__{architecture}__{scheme}__"
                                f"{optimizer.lower()}"
                            ),
                        }
                    )
    return rows


def ordered_cells(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return row-major cell IDs in expanding-square execution order."""

    conv = [float(value) for value in study["rho_search"]["rho_conv"]]
    dense = [float(value) for value in study["rho_search"]["rho_dense"]]
    cells = [
        {
            "index": conv_index * len(dense) + dense_index,
            "conv_axis_index": conv_index,
            "dense_axis_index": dense_index,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
        }
        for conv_index, rho_conv in enumerate(conv)
        for dense_index, rho_dense in enumerate(dense)
    ]
    cells.sort(
        key=lambda row: (
            max(row["conv_axis_index"], row["dense_axis_index"]),
            row["conv_axis_index"] + row["dense_axis_index"],
            row["conv_axis_index"],
            row["dense_axis_index"],
        )
    )
    return [dict(row, execution_rank=rank) for rank, row in enumerate(cells)]


def _resolved_output_root(study: Mapping[str, Any], value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    configured = Path(study["execution"]["result_root"])
    return (REPO_ROOT / configured).resolve()


def _materialize(
    study_path: Path,
    study: Mapping[str, Any],
    parent_path: Path,
    output_root: Path,
    *,
    target: str,
    device: str,
    dataset_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schema_version": RESOLVED_SCHEMA,
        "study_id": study["study_id"],
        "evidence_class": study["evidence_class"],
        "study_config": str(study_path),
        "study_config_sha256": _sha256_file(study_path),
        "parent_study_config": str(parent_path),
        "parent_study_config_sha256": _sha256_file(parent_path),
        "initializer_checkpoint_sha256": EXPECTED_INITIALIZER_SHA256,
        "code": _git_state(),
        "target": target,
        "device": device,
        "dataset_root": str(dataset_root),
        "surface_count": len(surface_specs(study)),
        "cells_per_surface": len(ordered_cells(study)),
        "surfaces": surface_specs(study),
        "official_test_read": False,
        "restart_interrupted_cells": True,
    }
    path = output_root / "study.resolved.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise RuntimeError(f"Existing resolved study does not match: {path}.")
    else:
        _write_json(path, resolved)
    return resolved


def _rho_args(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    source_config: Path,
    rho_root: Path,
    *,
    device: str,
    target: str,
    index: int | None,
    probe_only: bool = False,
    collect_only: bool = False,
    smoke: bool = False,
) -> Namespace:
    search = study["rho_search"]
    safety = search["safety"]
    return Namespace(
        config=str(source_config),
        output_root=str(rho_root),
        rho_conv=[float(value) for value in search["rho_conv"]],
        rho_dense=[float(value) for value in search["rho_dense"]],
        optimizer=surface["optimizer"],
        evidence_class=study["evidence_class"],
        bias_policy="zero",
        probe_batches=[2] if smoke else [128],
        stability_tolerance=1e9 if smoke else 0.5,
        epochs=1 if smoke else 3,
        checkpoint_every_epoch=False,
        post_candidate_gate_name=None,
        post_candidate_callback=None,
        max_batches=1 if smoke else None,
        max_validation_batches=1 if smoke else None,
        validation_batch_size=int(study["validation_batch_size"]),
        split_seed=int(study["split_seed"]),
        shuffle_seed=int(study["shuffle_seed"]),
        device=device,
        study_id=study["study_id"] + ("-smoke" if smoke else ""),
        target=target,
        probe_only=probe_only,
        canary_only=False,
        canary_steps=0,
        skip_canary=True,
        restart_interrupted_cells=True,
        expected_candidate_steps=1 if smoke else 10314,
        minimum_validation_accuracy=0.0,
        safety_warmup_steps=int(safety["warmup_steps"]),
        safety_ema_decay=float(safety["loss_ema_decay"]),
        safety_loss_factor=float(safety["loss_ema_factor"]),
        safety_gradient_factor=float(safety["gradient_rms_factor"]),
        safety_persistence=int(safety["persistence_steps"]),
        safety_bound_occupancy_increase_maximum=None,
        safety_projection_efficiency_minimum=None,
        safety_zero_proposal_epsilon=float(safety["zero_proposal_epsilon"]),
        safety_boundary_persistence=int(safety["boundary_persistence_steps"]),
        disable_safety_rejections=True,
        index=index,
        collect_only=collect_only,
        force=False,
        dry_run=False,
        smoke=smoke,
        reporting_command={
            "module": "experiments.run_conv3_exploratory_lr_ladder",
            "command": "smoke" if smoke else "run-surface",
            "surface_index": surface["index"],
            "cell_index": index,
        },
    )


def _cell_dir(rho_root: Path, cell: Mapping[str, Any]) -> Path:
    return rho_root / "cells" / (
        f"{int(cell['index']):03d}_rc_{_float_token(float(cell['rho_conv']))}_"
        f"rd_{_float_token(float(cell['rho_dense']))}"
    )


def _candidate_record(rho_root: Path, cell: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = _cell_dir(rho_root, cell)
    cell_path = directory / "cell.json"
    if not cell_path.is_file():
        return None
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    record = {
        **dict(cell),
        "cell_id": directory.name,
        "status": payload.get("status"),
        "safety_failure": payload.get("safety_failure"),
        "learning_rates_by_parameter": payload.get("learning_rates_by_parameter"),
        "path": str(directory),
        "final_validation_loss": None,
        "final_validation_accuracy": None,
        "median_projection_efficiency": None,
    }
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record["final_validation_loss"] = metrics.get("final_test_loss")
        record["final_validation_accuracy"] = metrics.get("final_test_accuracy")
    diagnostics_path = directory / "safety_diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        record["median_projection_efficiency"] = diagnostics.get(
            "median_projection_efficiency"
        )
    return record


def _is_allowed_terminal(record: Mapping[str, Any]) -> bool:
    status = record.get("status")
    if status in TERMINAL_CELL_STATUSES:
        return True
    # The lower-level transition monitor reports a non-finite diagnostic as a
    # SafetyRejection so it can stop immediately.  In this exploratory
    # protocol that is still a non-finite terminal, not a scientific safety
    # rejection.
    failure = record.get("safety_failure")
    return bool(
        status == "candidate_rejected_safety"
        and isinstance(failure, Mapping)
        and failure.get("kind") == "nonfinite_diagnostic"
    )


def _write_surface_snapshot(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    surface_dir: Path,
    *,
    state: str,
    error: str | None = None,
) -> dict[str, Any]:
    rho_root = surface_dir / "rho"
    candidates = [
        record
        for cell in ordered_cells(study)
        if (record := _candidate_record(rho_root, cell)) is not None
    ]
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate.get("status"))
        counts[key] = counts.get(key, 0) + 1
    terminal_count = sum(_is_allowed_terminal(candidate) for candidate in candidates)
    next_cell = next(
        (
            cell
            for cell in ordered_cells(study)
            if not (_cell_dir(rho_root, cell) / "cell.json").is_file()
        ),
        None,
    )
    numeric = [
        row for row in candidates if row.get("final_validation_accuracy") is not None
    ]
    best = (
        max(
            numeric,
            key=lambda row: (
                float(row["final_validation_accuracy"]),
                -float(row["final_validation_loss"]),
                -int(row["execution_rank"]),
            ),
        )
        if numeric
        else None
    )
    updated_at = _now()
    progress = {
        "schema_version": SURFACE_PROGRESS_SCHEMA,
        "study_id": study["study_id"],
        "surface": dict(surface),
        "state": state,
        "expected_cells": 64,
        "observed_cells": len(candidates),
        "terminal_cells": terminal_count,
        "status_counts": counts,
        "next_cell": next_cell,
        "updated_at": updated_at,
    }
    status = {
        "schema_version": SURFACE_STATUS_SCHEMA,
        "study_id": study["study_id"],
        "surface_id": surface["surface_id"],
        "status": state,
        "terminal": state in {"complete", "unresolved_probe", "failed"},
        "error": error,
        "progress": {
            "terminal_cells": terminal_count,
            "expected_cells": 64,
        },
        "updated_at": updated_at,
        "official_test_read": False,
    }
    summary = {
        "schema_version": SURFACE_SUMMARY_SCHEMA,
        "study_id": study["study_id"],
        "surface": dict(surface),
        "status": state,
        "complete": state == "complete" and terminal_count == 64,
        "expected_cells": 64,
        "terminal_cells": terminal_count,
        "status_counts": counts,
        "best_observation": best,
        "candidates": candidates,
        "error": error,
        "official_test_read": False,
    }
    _write_json(surface_dir / "progress.json", progress)
    _write_json(surface_dir / "status.json", status)
    _write_json(surface_dir / "summary.json", summary)
    return summary


def _prepare_surface(
    study: Mapping[str, Any],
    parent: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    *,
    device: str,
    dataset_root: Path,
) -> tuple[Path, Path]:
    checkpoint, asset = ensure_asset(
        parent,
        output_root,
        initializer="bounded_uniform",
        architecture="conv3",
        device=device,
        dataset_root=dataset_root,
    )
    if asset.get("checkpoint_sha256") != EXPECTED_INITIALIZER_SHA256:
        raise RuntimeError("The exploratory surface did not receive the pinned initializer.")
    source = build_source_config(
        parent,
        initializer="bounded_uniform",
        architecture="conv3",
        scheme=str(surface["scheme"]),
        optimizer=str(surface["optimizer"]),
        init_checkpoint_path=checkpoint,
        dataset_root=dataset_root,
    )
    surface_dir = output_root / "surfaces" / str(surface["surface_id"])
    source_path = surface_dir / "source_config.json"
    _write_json(source_path, source)
    return surface_dir, source_path


def run_surface(
    study: Mapping[str, Any],
    parent: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    *,
    device: str,
    target: str,
    dataset_root: Path,
) -> dict[str, Any]:
    surface_dir, source_path = _prepare_surface(
        study,
        parent,
        output_root,
        surface,
        device=device,
        dataset_root=dataset_root,
    )
    rho_root = surface_dir / "rho"
    _write_surface_snapshot(study, surface, surface_dir, state="running")
    try:
        probe = run_rho_search(
            _rho_args(
                study,
                surface,
                source_path,
                rho_root,
                device=device,
                target=target,
                index=None,
                probe_only=True,
            )
        )
        if probe.get("status") != "complete":
            return _write_surface_snapshot(
                study,
                surface,
                surface_dir,
                state="unresolved_probe",
                error=f"Optimizer probe status: {probe.get('status')!r}",
            )
        for cell in ordered_cells(study):
            run_rho_search(
                _rho_args(
                    study,
                    surface,
                    source_path,
                    rho_root,
                    device=device,
                    target=target,
                    index=int(cell["index"]),
                )
            )
            record = _candidate_record(rho_root, cell)
            if record is None or not _is_allowed_terminal(record):
                raise RuntimeError(
                    "Exploratory rho cell did not reach an allowed terminal state: "
                    f"cell={cell['index']}, record={record!r}."
                )
            _write_surface_snapshot(study, surface, surface_dir, state="running")
        run_rho_search(
            _rho_args(
                study,
                surface,
                source_path,
                rho_root,
                device=device,
                target=target,
                index=None,
                collect_only=True,
            )
        )
    except BaseException as error:
        _write_surface_snapshot(
            study, surface, surface_dir, state="failed", error=str(error)
        )
        raise
    return _write_surface_snapshot(study, surface, surface_dir, state="complete")


def run_smoke(
    study: Mapping[str, Any],
    parent: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    *,
    device: str,
    target: str,
    dataset_root: Path,
) -> dict[str, Any]:
    smoke_root = output_root / "smoke"
    surface_dir, source_path = _prepare_surface(
        study,
        parent,
        smoke_root,
        surface,
        device=device,
        dataset_root=dataset_root,
    )
    rho_root = surface_dir / "rho"
    probe = run_rho_search(
        _rho_args(
            study,
            surface,
            source_path,
            rho_root,
            device=device,
            target=target,
            index=None,
            probe_only=True,
            smoke=True,
        )
    )
    if probe.get("status") != "complete":
        result = {
            "schema_version": SMOKE_SCHEMA,
            "status": "unresolved_probe",
            "surface": dict(surface),
            "probe": probe,
            "official_test_read": False,
        }
        _write_json(smoke_root / "result.json", result)
        return result
    lowest = next(cell for cell in ordered_cells(study) if cell["index"] == 0)
    run_rho_search(
        _rho_args(
            study,
            surface,
            source_path,
            rho_root,
            device=device,
            target=target,
            index=0,
            smoke=True,
        )
    )
    candidate = _candidate_record(rho_root, lowest)
    if candidate is None or not _is_allowed_terminal(candidate):
        raise RuntimeError(f"Exploratory smoke cell did not complete: {candidate!r}.")
    result = {
        "schema_version": SMOKE_SCHEMA,
        "status": "complete",
        "surface": dict(surface),
        "candidate": candidate,
        "max_training_batches": 1,
        "max_validation_batches": 1,
        "skip_canary": True,
        "safety_rejections_enabled": False,
        "official_test_read": False,
    }
    _write_json(smoke_root / "result.json", result)
    return result


def collect(study: Mapping[str, Any], output_root: Path, *, write: bool) -> dict[str, Any]:
    surfaces = []
    counts: dict[str, int] = {}
    for surface in surface_specs(study):
        surface_dir = output_root / "surfaces" / surface["surface_id"]
        status_path = surface_dir / "status.json"
        summary_path = surface_dir / "summary.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = str(status.get("status"))
        else:
            state = "pending"
        counts[state] = counts.get(state, 0) + 1
        surfaces.append(
            {
                **surface,
                "status": state,
                "status_path": str(status_path),
                "summary_path": str(summary_path),
                "summary_present": summary_path.is_file(),
            }
        )
    result = {
        "schema_version": STUDY_SUMMARY_SCHEMA,
        "study_id": study["study_id"],
        "status": "complete" if counts.get("complete") == len(surfaces) else "partial",
        "counts": counts,
        "expected_surfaces": len(surfaces),
        "expected_cells_per_surface": 64,
        "surfaces": surfaces,
        "official_test_read": False,
    }
    if write:
        _write_json(output_root / "summary.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "smoke", "run-surface", "status", "collect")
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--target")
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_path, study, parent_path, parent = load_study(args.study)
    output_root = _resolved_output_root(study, args.output_root)
    surfaces = surface_specs(study)
    if not 0 <= args.surface_index < len(surfaces):
        raise ValueError(
            f"Expected --surface-index in [0,{len(surfaces) - 1}], "
            f"got {args.surface_index}."
        )
    device = args.device or str(study["execution"]["device"])
    target = args.target or str(study["execution"]["target"])
    dataset_root = Path(
        parent["dataset"]["root"] if args.dataset_root is None else args.dataset_root
    ).expanduser().resolve()
    if args.command in {"smoke", "run-surface"} and args.target is None:
        raise ValueError("Executing the multi-target ladder requires --target.")

    if args.command == "plan":
        result = {
            "schema_version": STUDY_SCHEMA,
            "study_id": study["study_id"],
            "study": str(study_path),
            "parent_study": str(parent_path),
            "output_root": str(output_root),
            "target": target,
            "device": device,
            "dataset_root": str(dataset_root),
            "surfaces": surfaces,
            "rho_conv": study["rho_search"]["rho_conv"],
            "rho_dense": study["rho_search"]["rho_dense"],
            "execution_order": ordered_cells(study),
            "cells_per_surface": 64,
            "total_cells": 64 * len(surfaces),
            "candidate_epochs": 3,
            "skip_canary": True,
            "safety_rejections_enabled": False,
        }
    elif args.command == "smoke":
        _materialize(
            study_path,
            study,
            parent_path,
            output_root / "smoke",
            target=target,
            device=device,
            dataset_root=dataset_root,
        )
        result = run_smoke(
            study,
            parent,
            output_root,
            surfaces[args.surface_index],
            device=device,
            target=target,
            dataset_root=dataset_root,
        )
    elif args.command == "run-surface":
        _materialize(
            study_path,
            study,
            parent_path,
            output_root,
            target=target,
            device=device,
            dataset_root=dataset_root,
        )
        result = run_surface(
            study,
            parent,
            output_root,
            surfaces[args.surface_index],
            device=device,
            target=target,
            dataset_root=dataset_root,
        )
    elif args.command == "status":
        result = collect(study, output_root, write=False)
    elif args.command == "collect":
        result = collect(study, output_root, write=True)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
