#!/usr/bin/env python3
"""Run the directed, non-duplicate Conv1/Conv2 exploratory LR search.

The study keeps the exact bounded-uniform initializer and zero-bias contract
from the accepted seed-0 bounded search.  Each surface declares sparse
Cartesian blocks so that already completed exact-contract coordinates are not
repeated.  The run is intentionally exploratory: scientific safety rejection,
per-cell canaries, and post-training T/K checks are disabled, while non-finite
and operational failures remain terminal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    / "perfectdiode_conv12_directed_exploratory_lr_search_seed0_20260812_v1.json"
)
STUDY_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-study/v1"
RESOLVED_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-resolved/v1"
SURFACE_STATUS_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-status/v1"
SURFACE_PROGRESS_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-progress/v1"
SURFACE_SUMMARY_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-surface/v1"
STUDY_SUMMARY_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-summary/v1"
SMOKE_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-smoke/v1"
EXPECTED_PARENT_SCHEMA = "perfectdiode-conv123-bounded-uniform-zero-bias-rho-study/v1"
EXPECTED_PARENT_STUDY_ID = (
    "perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1"
)
EXPECTED_PARENT_SHA256 = (
    "ee8688a13470016d2cbb05561d2aa6d78d603f33f1c4b269252c3069bfac6a05"
)
EXPECTED_INITIALIZER_SHA256 = {
    "conv1": "235d7ae56a277e097e840a4eb11cf16e325ed2d3eb80c9055f7903493cb1f16e",
    "conv2": "6469bc8f465e9abd2b4720231ad03d0116fc176ef4e437dc005027204c4d88d2",
}
EXPECTED_TK = {"conv1": (4, 4), "conv2": (6, 6)}
TERMINAL_CELL_STATUSES = {"complete", "candidate_rejected_nonfinite"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_ladders() -> tuple[list[float], list[float]]:
    return (
        [0.009 * (3.0 ** (index - 7)) for index in range(11)],
        [0.03 * (3.0 ** (index - 7)) for index in range(10)],
    )


def _require_float_ladder(value: Any, expected: Sequence[float], label: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"Expected {label} to be a list.")
    observed = [float(item) for item in value]
    if len(observed) != len(expected) or any(
        not math.isclose(left, right, rel_tol=1e-15, abs_tol=0.0)
        for left, right in zip(observed, expected)
    ):
        raise ValueError(f"Unexpected {label}: {observed!r}.")
    return observed


def _surface_key(surface: Mapping[str, Any]) -> str:
    return f"{surface['architecture']}__{surface['scheme']}__{str(surface['optimizer']).lower()}"


def _block_pairs(
    blocks: Any,
    *,
    conv_count: int,
    dense_count: int,
    label: str,
) -> list[tuple[int, int]]:
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"Expected non-empty block list for {label}.")
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for block_index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise ValueError(f"Malformed block {block_index} for {label}.")
        conv = block.get("rho_conv_indices")
        dense = block.get("rho_dense_indices")
        if not isinstance(conv, list) or not conv or not isinstance(dense, list) or not dense:
            raise ValueError(f"Block {block_index} for {label} requires both index lists.")
        conv_indices = [int(value) for value in conv]
        dense_indices = [int(value) for value in dense]
        if len(set(conv_indices)) != len(conv_indices) or len(set(dense_indices)) != len(dense_indices):
            raise ValueError(f"Duplicate axis index in block {block_index} for {label}.")
        if any(value < 0 or value >= conv_count for value in conv_indices):
            raise ValueError(f"Conv index outside the ladder in block {block_index} for {label}.")
        if any(value < 0 or value >= dense_count for value in dense_indices):
            raise ValueError(f"Dense index outside the ladder in block {block_index} for {label}.")
        for conv_index in conv_indices:
            for dense_index in dense_indices:
                pair = (conv_index, dense_index)
                if pair in seen:
                    raise ValueError(f"Overlapping directed blocks repeat {pair} for {label}.")
                seen.add(pair)
                pairs.append(pair)
    return pairs


def load_study(
    path: str | Path = DEFAULT_STUDY,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Load and fail-closed validate the directed study and its parent."""

    source = Path(path).expanduser().resolve()
    study = json.loads(source.read_text(encoding="utf-8"))
    if study.get("schema_version") != STUDY_SCHEMA:
        raise ValueError(f"Unexpected directed-study schema: {study.get('schema_version')!r}.")
    if study.get("evidence_class") != "ordinary_mnist_exploratory":
        raise ValueError("The directed LR study must be ordinary-MNIST exploratory evidence.")
    expected_scope = {
        "architectures": ["conv1", "conv2"],
        "schemes": ["baseline", "ours", "legacy"],
        "optimizers": ["SGD", "Adam"],
        "initializers": ["bounded_uniform"],
        "excluded": ["conv3"],
    }
    if study.get("scope") != expected_scope:
        raise ValueError(f"Expected scope {expected_scope!r}.")

    parent_record = study.get("parent")
    if not isinstance(parent_record, Mapping):
        raise ValueError("The directed study requires a parent record.")
    parent_relative = parent_record.get("study_config")
    if not isinstance(parent_relative, str) or not parent_relative:
        raise ValueError("The parent study_config must be a repository-relative path.")
    parent_path = (REPO_ROOT / parent_relative).resolve()
    if not parent_path.is_relative_to(REPO_ROOT):
        raise ValueError("The parent config must stay beneath the repository.")
    if parent_record.get("study_id") != EXPECTED_PARENT_STUDY_ID:
        raise ValueError("Unexpected parent study identity.")
    if parent_record.get("study_config_sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("Unexpected pinned parent-config SHA-256.")
    if _sha256_file(parent_path) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("The current parent config does not match its pinned SHA-256.")
    loaded_parent_path, parent = load_parent_study(parent_path)
    if loaded_parent_path != parent_path:
        raise RuntimeError("The parent loader resolved a different config path.")
    if parent.get("schema_version") != EXPECTED_PARENT_SCHEMA:
        raise ValueError("Unexpected parent study schema.")
    if parent.get("study_id") != EXPECTED_PARENT_STUDY_ID:
        raise ValueError("Unexpected loaded parent study identity.")
    if parent_record.get("initializer_checkpoint_sha256_by_architecture") != EXPECTED_INITIALIZER_SHA256:
        raise ValueError("Unexpected architecture-shared initializer hashes.")

    model = parent["model"]
    if [float(model["weight_min"]), float(model["weight_max"])] != [1e-5, 1e-4]:
        raise ValueError("The parent weight interval is not [1e-5,1e-4].")
    for architecture, expected_tk in EXPECTED_TK.items():
        record = model["architectures"][architecture]
        if (int(record["T"]), int(record["K"])) != expected_tk:
            raise ValueError(f"Unexpected accepted T/K for {architecture}.")
    bias = parent.get("bias_contract")
    if not isinstance(bias, Mapping) or float(bias.get("learning_rate", -1.0)) != 0.0:
        raise ValueError("The parent does not retain exact-zero bias learning rates.")

    conv_expected, dense_expected = _expected_ladders()
    search = study.get("rho_search")
    if not isinstance(search, Mapping):
        raise ValueError("The directed study requires rho_search.")
    conv = _require_float_ladder(search.get("rho_conv_ladder"), conv_expected, "rho_conv_ladder")
    dense = _require_float_ladder(search.get("rho_dense_ladder"), dense_expected, "rho_dense_ladder")
    required_search = {
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
            raise ValueError(f"Expected rho_search.{key}={expected!r}.")
    safety = search.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("The directed study requires explicit safety diagnostics.")
    if safety.get("bound_occupancy_increase_maximum") is not None:
        raise ValueError("Bound occupancy must remain diagnostic-only.")
    if safety.get("projection_efficiency_minimum") is not None:
        raise ValueError("Projection efficiency must remain diagnostic-only.")

    surfaces = study.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) != 12:
        raise ValueError("Expected exactly twelve directed surfaces.")
    expected_combinations = {
        (architecture, scheme, optimizer)
        for architecture in expected_scope["architectures"]
        for scheme in expected_scope["schemes"]
        for optimizer in expected_scope["optimizers"]
    }
    observed_combinations: set[tuple[str, str, str]] = set()
    observed_keys: set[str] = set()
    total_cells = 0
    prior = study.get("prior_numeric_blocks")
    if not isinstance(prior, Mapping):
        raise ValueError("The directed study requires prior_numeric_blocks.")
    retries = study.get("retry_non_numeric_coordinates", {})
    if not isinstance(retries, Mapping):
        raise ValueError("retry_non_numeric_coordinates must be a mapping.")
    for expected_index, surface in enumerate(surfaces):
        if not isinstance(surface, Mapping) or int(surface.get("index", -1)) != expected_index:
            raise ValueError("Surface indices must be consecutive and match list order.")
        combination = (
            str(surface.get("architecture")),
            str(surface.get("scheme")),
            str(surface.get("optimizer")),
        )
        if combination not in expected_combinations or combination in observed_combinations:
            raise ValueError(f"Unexpected or duplicate surface combination: {combination!r}.")
        observed_combinations.add(combination)
        key = _surface_key(surface)
        observed_keys.add(key)
        pairs = _block_pairs(
            surface.get("blocks"),
            conv_count=len(conv),
            dense_count=len(dense),
            label=key,
        )
        if len(pairs) != int(surface.get("expected_cells", -1)):
            raise ValueError(f"Expected-cell count mismatch for {key}.")
        prior_pairs = set(
            _block_pairs(
                prior.get(key),
                conv_count=len(conv),
                dense_count=len(dense),
                label=f"prior:{key}",
            )
        )
        overlap = prior_pairs.intersection(pairs)
        if overlap:
            raise ValueError(f"Directed surface repeats prior numeric pairs for {key}: {sorted(overlap)!r}.")
        retry_pairs = {tuple(int(item) for item in pair) for pair in retries.get(key, [])}
        if any(len(pair) != 2 for pair in retries.get(key, [])):
            raise ValueError(f"Malformed retry coordinate for {key}.")
        if not retry_pairs.issubset(set(pairs)) or retry_pairs.intersection(prior_pairs):
            raise ValueError(f"Invalid non-numeric retry declaration for {key}.")
        total_cells += len(pairs)
    if observed_combinations != expected_combinations or set(prior) != observed_keys:
        raise ValueError("Surface/prior coverage is not exactly the declared 2x3x2 scope.")
    execution = study.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("The directed study requires execution metadata.")
    if int(execution.get("total_new_cells", -1)) != total_cells or total_cells != 257:
        raise ValueError(f"Expected the directed study to contain 257 cells, got {total_cells}.")
    if int(execution.get("candidate_epoch_budget", -1)) != total_cells * 3:
        raise ValueError("Candidate-epoch budget does not match the directed cells.")
    if int(execution.get("expected_optimizer_steps", -1)) != total_cells * 10314:
        raise ValueError("Optimizer-step budget does not match the directed cells.")
    return source, study, parent_path, parent


def surface_specs(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in study["surfaces"]:
        row = dict(raw)
        row["initializer"] = "bounded_uniform"
        row["surface_id"] = (
            f"bounded_uniform__{row['architecture']}__{row['scheme']}__"
            f"{str(row['optimizer']).lower()}"
        )
        rows.append(row)
    return rows


def surface_cells(study: Mapping[str, Any], surface: Mapping[str, Any]) -> list[dict[str, Any]]:
    conv_ladder = [float(value) for value in study["rho_search"]["rho_conv_ladder"]]
    dense_ladder = [float(value) for value in study["rho_search"]["rho_dense_ladder"]]
    pairs = _block_pairs(
        surface["blocks"],
        conv_count=len(conv_ladder),
        dense_count=len(dense_ladder),
        label=str(surface["surface_id"]),
    )
    conv_indices = sorted({pair[0] for pair in pairs})
    dense_indices = sorted({pair[1] for pair in pairs})
    conv_position = {value: index for index, value in enumerate(conv_indices)}
    dense_position = {value: index for index, value in enumerate(dense_indices)}
    rows = []
    for execution_rank, (conv_index, dense_index) in enumerate(pairs):
        rows.append(
            {
                "index": conv_position[conv_index] * len(dense_indices) + dense_position[dense_index],
                "execution_rank": execution_rank,
                "rho_conv_ladder_index": conv_index,
                "rho_dense_ladder_index": dense_index,
                "rho_conv": conv_ladder[conv_index],
                "rho_dense": dense_ladder[dense_index],
            }
        )
    if len({row["index"] for row in rows}) != len(rows):
        raise RuntimeError(f"Sparse Cartesian indexing collided for {surface['surface_id']}.")
    return rows


def surface_axes(study: Mapping[str, Any], surface: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    cells = surface_cells(study, surface)
    conv = sorted({float(row["rho_conv"]) for row in cells})
    dense = sorted({float(row["rho_dense"]) for row in cells})
    return conv, dense


def _resolved_output_root(study: Mapping[str, Any], value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    return (REPO_ROOT / str(study["execution"]["result_root"])).resolve()


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
        "initializer_checkpoint_sha256_by_architecture": EXPECTED_INITIALIZER_SHA256,
        "code": _git_state(),
        "target": target,
        "device": device,
        "dataset_root": str(dataset_root),
        "surface_count": len(surface_specs(study)),
        "total_cells": sum(len(surface_cells(study, row)) for row in surface_specs(study)),
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
    smoke: bool = False,
) -> Namespace:
    search = study["rho_search"]
    safety = search["safety"]
    conv, dense = surface_axes(study, surface)
    return Namespace(
        config=str(source_config),
        output_root=str(rho_root),
        rho_conv=conv,
        rho_dense=dense,
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
        collect_only=False,
        force=False,
        dry_run=False,
        smoke=smoke,
        reporting_command={
            "module": "experiments.run_conv12_directed_exploratory_lr_search",
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
        record["median_projection_efficiency"] = diagnostics.get("median_projection_efficiency")
    return record


def _is_allowed_terminal(record: Mapping[str, Any]) -> bool:
    status = record.get("status")
    if status in TERMINAL_CELL_STATUSES:
        return True
    failure = record.get("safety_failure")
    return bool(
        status == "candidate_rejected_safety"
        and isinstance(failure, Mapping)
        and failure.get("kind") in {"nonfinite_diagnostic", "nonfinite_training_value"}
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
    cells = surface_cells(study, surface)
    candidates = [
        record
        for cell in cells
        if (record := _candidate_record(rho_root, cell)) is not None
    ]
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate.get("status"))
        counts[key] = counts.get(key, 0) + 1
    terminal_count = sum(_is_allowed_terminal(candidate) for candidate in candidates)
    next_cell = next(
        (cell for cell in cells if not (_cell_dir(rho_root, cell) / "cell.json").is_file()),
        None,
    )
    numeric = [row for row in candidates if row.get("final_validation_accuracy") is not None]
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
    expected_cells = len(cells)
    updated_at = _now()
    progress = {
        "schema_version": SURFACE_PROGRESS_SCHEMA,
        "study_id": study["study_id"],
        "surface": dict(surface),
        "state": state,
        "expected_cells": expected_cells,
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
        "progress": {"terminal_cells": terminal_count, "expected_cells": expected_cells},
        "updated_at": updated_at,
        "official_test_read": False,
    }
    summary = {
        "schema_version": SURFACE_SUMMARY_SCHEMA,
        "study_id": study["study_id"],
        "surface": dict(surface),
        "status": state,
        "complete": state == "complete" and terminal_count == expected_cells,
        "expected_cells": expected_cells,
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
    architecture = str(surface["architecture"])
    checkpoint, asset = ensure_asset(
        parent,
        output_root,
        initializer="bounded_uniform",
        architecture=architecture,
        device=device,
        dataset_root=dataset_root,
    )
    expected_sha = EXPECTED_INITIALIZER_SHA256[architecture]
    if asset.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError(f"The {architecture} surface did not receive its pinned initializer.")
    source = build_source_config(
        parent,
        initializer="bounded_uniform",
        architecture=architecture,
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
        study, parent, output_root, surface, device=device, dataset_root=dataset_root
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
        for cell in surface_cells(study, surface):
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
                    "Directed rho cell did not reach an allowed terminal state: "
                    f"cell={cell['index']}, record={record!r}."
                )
            _write_surface_snapshot(study, surface, surface_dir, state="running")
    except BaseException as error:
        _write_surface_snapshot(study, surface, surface_dir, state="failed", error=str(error))
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
        study, parent, smoke_root, surface, device=device, dataset_root=dataset_root
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
    first = surface_cells(study, surface)[0]
    run_rho_search(
        _rho_args(
            study,
            surface,
            source_path,
            rho_root,
            device=device,
            target=target,
            index=int(first["index"]),
            smoke=True,
        )
    )
    candidate = _candidate_record(rho_root, first)
    if candidate is None or not _is_allowed_terminal(candidate):
        raise RuntimeError(f"Directed smoke cell did not complete: {candidate!r}.")
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


def collect(study: Mapping[str, Any], roots: Iterable[Path], *, write: Path | None) -> dict[str, Any]:
    surfaces = surface_specs(study)
    discovered: dict[str, dict[str, Any]] = {}
    for root in roots:
        for surface in surfaces:
            directory = root.expanduser().resolve() / "surfaces" / str(surface["surface_id"])
            status_path = directory / "status.json"
            summary_path = directory / "summary.json"
            if not status_path.is_file():
                continue
            if surface["surface_id"] in discovered:
                raise RuntimeError(f"Duplicate surface authority: {surface['surface_id']}.")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            discovered[str(surface["surface_id"])] = {
                **surface,
                "status": status.get("status"),
                "root": str(root.expanduser().resolve()),
                "status_path": str(status_path),
                "summary_path": str(summary_path),
                "summary_present": summary_path.is_file(),
            }
    rows = [
        discovered.get(
            str(surface["surface_id"]),
            {**surface, "status": "pending", "summary_present": False},
        )
        for surface in surfaces
    ]
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row["status"])
        counts[state] = counts.get(state, 0) + 1
    result = {
        "schema_version": STUDY_SUMMARY_SCHEMA,
        "study_id": study["study_id"],
        "status": "complete" if counts.get("complete") == len(surfaces) else "partial",
        "counts": counts,
        "expected_surfaces": len(surfaces),
        "expected_cells": sum(len(surface_cells(study, row)) for row in surfaces),
        "surfaces": rows,
        "official_test_read": False,
    }
    if write is not None:
        _write_json(write.expanduser().resolve(), result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "run-surface", "status", "collect"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--target")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_path, study, parent_path, parent = load_study(args.study)
    output_root = _resolved_output_root(study, args.output_root)
    surfaces = surface_specs(study)
    if not 0 <= args.surface_index < len(surfaces):
        raise ValueError(
            f"Expected --surface-index in [0,{len(surfaces) - 1}], got {args.surface_index}."
        )
    device = args.device or str(study["execution"]["device"])
    target = args.target or str(surfaces[args.surface_index]["target"])
    dataset_root = Path(
        parent["dataset"]["root"] if args.dataset_root is None else args.dataset_root
    ).expanduser().resolve()
    if args.command in {"smoke", "run-surface"} and args.target is None:
        raise ValueError("Executing the multi-target study requires --target.")

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
            "surfaces": [
                {**surface, "cells": surface_cells(study, surface)} for surface in surfaces
            ],
            "total_cells": sum(len(surface_cells(study, row)) for row in surfaces),
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
    elif args.command in {"status", "collect"}:
        roots = args.root or [output_root]
        result = collect(
            study,
            roots,
            write=args.summary_json if args.command == "collect" else None,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
